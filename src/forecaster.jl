using DataFrames
using Dates
using Stan
using OrderedCollections
using Statistics
using Gadfly
using Plotly

include("make_holidays.jl")


mutable struct Prophet
    growth::Union{String, Nothing}
    changepoints::Union{Array{Date,1}, Nothing}
    n_changepoints::Int
    changepoint_range::Float64
    yearly_seasonality::Union{Bool, Int, String}
    weekly_seasonality::Union{Bool, Int, String}
    daily_seasonality::Union{Bool, Int, String}
    holidays::Union{DataFrame, Nothing}
    seasonality_mode::String
    seasonality_prior_scale::Float64
    holidays_prior_scale::Float64
    changepoint_prior_scale::Float64
    mcmc_samples::Int
    interval_width::Float64
    uncertainty_samples::Int
    stan_backend::Union{String, Nothing}
    start::Union{Date, Nothing}
    y_scale::Union{Float64, Nothing}
    logistic_floor::Bool
    t_scale::Union{Period, Nothing}
    changepoints_t::Union{Array{Float64,1}, Nothing}
    seasonality::Dict{String, Function}
    extra_regressors::Dict{String, Dict{String, Union{String, Float64}}}
    country_holidays::Union{String, Nothing}
    stan_fit::Union{Any, Nothing}
    params::Dict{String, Array{Float64,2}}
    history::Union{DataFrame, Nothing}
    history_dates::Union{Array{Date,1}, Nothing}
    train_component_cols::Union{DataFrame, Nothing}
    component_modes::Dict{String, String}
    train_holiday_names::Union{Array{String,1}, Nothing}
end

function Prophet(;
        growth="linear",
        changepoints=nothing,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=nothing,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=nothing,
        country_holidays=nothing
    )
    return Prophet(
        growth,
        changepoints,
        n_changepoints,
        changepoint_range,
        yearly_seasonality,
        weekly_seasonality,
        daily_seasonality,
        holidays,
        seasonality_mode,
        seasonality_prior_scale,
        holidays_prior_scale,
        changepoint_prior_scale,
        mcmc_samples,
        interval_width,
        uncertainty_samples,
        stan_backend,
        nothing, # start
        nothing, # y_scale
        false, # logistic_floor
        nothing, # t_scale
        nothing, # changepoints_t
        Dict{String, Function}(), # seasonality
        Dict{String, Dict{String, Union{String, Float64}}}(), # extra_regressors
        country_holidays,
        nothing, # stan_fit
        Dict{String, Array{Float64, 2}}(), # params
        nothing, # history
        nothing, # history_dates
        nothing, # train_component_cols
        Dict{String, String}(), # component_modes
        nothing, # train_holiday_names
    )
end

function _load_stan_backend(model::Prophet)
    if model.stan_backend == "CMDSTAN"
        return Stan
    else
        error("Unrecognized stan_backend: $(model.stan_backend)")
    end
end

function validate_inputs(model::Prophet)
    if model.growth != "linear" && model.growth != "logistic"
        error("Invalid growth: '$(model.growth)'; must be 'linear' or 'logistic'.")
    end

    if model.changepoints !== nothing
        if model.n_changepoints != length(model.changepoints)
            error("Number of changepoints must be equal to the length of the changepoints provided.")
        end
    else
        if model.n_changepoints <= 0
            error("Number of changepoints must be greater than 0.")
        end
    end

    if model.mcmc_samples < 0
        error("mcmc_samples must be greater than or equal to 0.")
    end

    if model.interval_width < 0 || model.interval_width > 1
        error("interval_width must be between 0 and 1.")
    end

    if model.uncertainty_samples < 0
        error("uncertainty_samples must be greater than or equal to 0.")
    end

    if model.changepoint_range < 0 || model.changepoint_range > 1
        error("changepoint_range must be between 0 and 1.")
    end

    if model.seasonality_mode != "additive" && model.seasonality_mode != "multiplicative"
        error("Invalid seasonality_mode: '$(model.seasonality_mode)'; must be 'additive' or 'multiplicative'.")
    end
end


function _set_changepoints(model::Prophet)
    if model.specified_changepoints
        model.changepoints .= model.specified_changepoints
    else
        potential_changepoints = range(model.start, stop=model.t_scale, length=model.n_changepoints + 1)[1:end-1]
        model.changepoints .= potential_changepoints
    end
    return model
end


function _get_changepoint_matrix(t::Vector{Float64}, ts::Vector{Float64})
    A = zeros(length(t), length(ts))
    for i in 1:length(t)
        for j in 1:length(ts)
            if t[i] >= ts[j]
                A[i, j] = 1
            end
        end
    end
    return A
end


function _fourier_series(t::Vector{Float64}, p::Float64, n::Int64)
    x = hcat([[sin(2 * pi * i * t / p), cos(2 * pi * i * t / p)] for i in 1:n]...)
    return x
end


function _seasonality_matrix(t::Vector{Float64}, p::Vector{Float64}, n::Int64)
    out = hcat([_fourier_series(t, p[i], n) for i in 1:length(p)]...)
    return out
end

function _linear_growth_init(df::DataFrame, model::Prophet)
    t_s = df[:,:] .- model.start
    A = _get_changepoint_matrix(t_s, model.changepoints .- model.start)
    k = model.growth_init
    m = model.growth_init * (1 - A[1, :]) * k
    return k, m
end


function _get_growth_scaler(prior_scale::Float64, t_s::Vector{Float64}, A::Matrix{Float64})
    delta_scale = prior_scale * (t_s[end] - t_s[1]) / sum(abs.(A * ones(size(A, 2))))
    return delta_scale
end


function _get_seasonality_scaler(prior_scale::Float64, t_s::Vector{Float64})
    beta_scale = prior_scale * mean(abs.(t_s))
    return beta_scale
end

function _get_auto_seasonalities(df::DataFrame)
    start_year = Dates.year(minimum(df[:ds]))
    end_year = Dates.year(maximum(df[:ds]))
    yearly = (start_year != end_year)
    return Dict("yearly" => yearly)
end

# predict function
function predict(self, df::DataFrame)
    seasonal_features, _, component_cols, _ = self.make_all_seasonality_features(df)

    if self.stan_fit == nothing
        error("Model has not been fit; fit the model first.")
    end

    trend = self.predict_trend(df)
    seasonal_components = seasonal_features * component_cols
    yhat = trend .+ self.seasonalities["yearly"].shape[1] == 0 ? seasonal_components : seasonal_components[:, 1:self.seasonalities["yearly"].shape[1]]
    for i in 1:length(self.seasonalities["yearly"].shape[1])
        yhat += seasonal_components[:, i]
    end
    return yhat
end

# predict_trend function
function predict_trend(self, df::DataFrame)
    k = self.params["k"]
    m = self.params["m"]
    delta = self.params["delta"]
    t = df[:]["t"]

    if self.growth == "linear"
        trend = k .* t .+ m
    elseif self.growth == "logistic"
        cap = df[:]["cap"]
        floor = df[:]["floor"]
        gamma = -1 .* k .* (cap .- floor)
        k_cumulative = cumsum(delta .* self.s_a) ./ self.s_a[1]
        A = 1 .- exp.(-1 .* (k_cumulative .- gamma .* t))
        B = 1 .+ exp.(-1 .* k_cumulative)
        trend = cap .- (cap .- floor) ./ (A ./ B)
    end
    return trend
end

# predict_seasonal function
function predict_seasonal(self, components::DataFrame)
    seasonal = deepcopy(components)
    for seasonality in keys(self.seasonalities)
        cols = [col for col in names(components) if startswith(col, seasonality)]
        seasonal[:, seasonality] = sum(seasonal[:, cols], dims=2)
    end
    return seasonal
end

# predict_uncertainty function
function predict_uncertainty(self, df::DataFrame)
    sim_values = simulation_smoother(self, df)
    lower_p = self.interval_width / 2
    upper_p = 1 - lower_p
    lower = DataFrame()
    upper = DataFrame()

    for sim in names(sim_values)
        lower[!, sim] = quantile(sim_values[:, sim], lower_p)
        upper[!, sim] = quantile(sim_values[:, sim], upper_p)
    end
    uncertainty = DataFrame()
    uncertainty[:, "lower"] = lower
    uncertainty[:, "upper"] = upper
    return uncertainty
end

# extend_dataframe function
function extend_dataframe(self, df::DataFrame, n::Int)
    last_date = df[end, :ds]
    dates = [last_date + Dates.Day(i) for i in 1:n]
    future = DataFrame(ds = dates)
    return vcat(df, future)
end

# predict_seasonal_components function
function predict_seasonal_components(self, df::DataFrame)
    seasonal_features, _ = self:make_all_seasonality_features(df)
    
    seasonal_components = Matrix{Float64}(undef, nrow(df), 0)

    for component in self.seasonalities
        features = seasonal_features[!, Symbol.(component.feature_names)]
        X = Matrix(features)
        component_vals = X * self.params[Symbol(component.name), :]
        seasonal_components = hcat(seasonal_components, component_vals)
    end
    
    seasonal_components = DataFrame(seasonal_components, Symbol.(self.seasonality_component_columns))
    return seasonal_components
end



function validate_inputs(self)
    if !(self.growth in ["linear", "logistic", "flat"])
        error("Parameter growth should be linear, logistic or flat.")
    end
    if !(isa(self.changepoint_range, Real) && 0 <= self.changepoint_range <= 1)
        error("Parameter changepoint_range must be in [0, 1]")
    end
    if !isnothing(self.holidays)
        if !(isa(self.holidays, DataFrame) && "ds" in names(self.holidays) && "holiday" in names(self.holidays))
            error("holidays must be a DataFrame with ds and holiday columns.")
        end
        self.holidays.ds = DateTime.(self.holidays.ds)
        if any(isna.(self.holidays.ds)) || any(isna.(self.holidays.holiday))
            error("Found a NaN in holidays dataframe.")
        end
        has_lower = "lower_window" in names(self.holidays)
        has_upper = "upper_window" in names(self.holidays)
        if has_lower != has_upper
            error("Holidays must have both lower_window and upper_window, or neither")
        end
        if has_lower
            if maximum(self.holidays.lower_window) > 0
                error("Holiday lower_window should be <= 0")
            end
            if minimum(self.holidays.upper_window) < 0
                error("Holiday upper_window should be >= 0")
            end
        end
        for h in unique(self.holidays.holiday)
            validate_column_name(self, h, false, true, true)
        end
    end
    if !(self.seasonality_mode in ["additive", "multiplicative"])
        error("seasonality_mode must be additive or multiplicative")
    end
end


function validate_column_name(self, name; check_holidays=true, check_seasonalities=true, check_regressors=true)
    if "_delim_" in name
        error("Name cannot contain '_delim_'")
    end
    reserved_names = [
        "trend", "additive_terms", "daily", "weekly", "yearly",
        "holidays", "zeros", "extra_regressors_additive", "yhat",
        "extra_regressors_multiplicative", "multiplicative_terms",
    ]
    append!(reserved_names, [n * "_lower" for n in reserved_names])
    append!(reserved_names, [n * "_upper" for n in reserved_names])
    append!(reserved_names, [
        "ds", "y", "cap", "floor", "y_scaled", "cap_scaled"])

    if name in reserved_names
        error("Name '$(name)' is reserved.")
    end
    if check_holidays && !isnothing(self.holidays) && name in unique(self.holidays.holiday)
        error("Name '$(name)' already used for a holiday.")
    end
    if check_holidays && !isnothing(self.country_holidays) && name in get_holiday_names(self.country_holidays)
        error("Name '$(name)' is a holiday name in $(self.country_holidays).")
    end
    if check_seasonalities && name in keys(self.seasonalities)
        error("Name '$(name)' already used for a seasonality.")
    end
    if check_regressors && name in keys(self.extra_regressors)
        error("Name '$(name)' already used for an added regressor.")
    end
end


function setup_dataframe(self, df, initialize_scales=false)
    if "y" in names(df)
        df.y = float.(df.y)
        if any(isinf.(df.y))
            error("Found infinity in column y.")
        end
    end
    if eltype(df.ds) == Int64
        df.ds = string.(df.ds)
    end
    df.ds = DateTime.(df.ds)
    if !isnothing(df.ds[1].zone)
        error("Column ds has timezone specified, which is not supported. Remove timezone.")
    end
    if any(isna.(df.ds))
        error("Found NaN in column ds.")
    end
    for name in keys(self.extra_regressors)
        if name ∉ names(df)
            error("Regressor '$(name)' missing from dataframe")
        end
        df[!, name] = float.(df[!, name])
        if any(isna.(df[!, name]))
            error("Found NaN in column '$(name)'")
        end
    end
    for (_, props) in self.seasonalities
        condition_name = props["condition_name"]
        if !isnothing(condition_name)
            if condition_name ∉ names(df)
                error("Condition '$(condition_name)' missing from dataframe")
            end
            if !all(x -> x in [true, false], df[!, condition_name])
                error("Found non-boolean in column '$(condition_name)'")
            end
            df[!, condition_name] = Bool.(df[!, condition_name])
        end
    end

    if names(df) == "ds"
        rename!(df, "ds" => nothing)
    end
    sort!(df, "ds")
    df = reset_index(df)

    initialize_scales(self, initialize_scales, df)

    if self.logistic_floor
        if "floor" ∉ names(df)
            error("Expected column floor.")
        end
    else
        df.floor = 0
    end
    if self.growth == "logistic"
        if "cap" ∉ names(df)
            error("Capacities must be supplied for logistic growth in column cap")
        end
        if any(df.cap .<= df.floor)
            error("cap must be greater than floor (which defaults to 0).")
        end
        df.cap_scaled = (df.cap .- df.floor) ./ self.y_scale
    end

    df.t = (df.ds .- self.start) ./ self.t_scale
    if "y" in names(df)
        df.y_scaled = (df.y .- df.floor) ./ self.y_scale
    end

    for (name, props) in self.extra_regressors
        df[!, name] = ((df[!, name] .- props["mu"]) ./ props["std"])
    end
    return df
end


function initialize_scales(self, initialize_scales, df)
    if !initialize_scales
        return
    end
    if self.growth == "logistic" && "floor" in names(df)
        self.logistic_floor = true
        floor = df.floor
    else
        floor = 0.
    end
    self.y_scale = float(maximum(abs.(df.y .- floor)))
    if self.y_scale == 0
        self.y_scale = 1.0
    end
    self.start = minimum(df.ds)
    self.t_scale = maximum(df.ds) - self.start
    for (name, props) in self.extra_regressors
        props["mu"] = mean(df[!, name])
        props["std"] = std(df[!, name])
    end
    for (_, props) in self.seasonalities
        props["mu"] = mean(df.t)
        props["std"] = std(df.t)
    end
end



function set_changepoints(self)
    if self.changepoints !== nothing
        if isempty(self.changepoints)
            pass
        else
            too_low = minimum(self.changepoints) < minimum(self.history.ds)
            too_high = maximum(self.changepoints) > maximum(self.history.ds)
            if too_low || too_high
                error("Changepoints must fall within training data.")
            end
        end
    else
        hist_size = floor(Int, self.history.nrow * self.changepoint_range)
        if self.n_changepoints + 1 > hist_size
            self.n_changepoints = hist_size - 1
            @info "n_changepoints greater than number of observations. Using $(self.n_changepoints)."
        end
        if self.n_changepoints > 0
            cp_indexes = round.(Int, range(0, stop=hist_size - 1, length=self.n_changepoints + 1))
            self.changepoints = self.history[cp_indexes, :ds][2:end]
        else
            self.changepoints = DateTime[]  # Empty changepoints
        end
    end
    if !isempty(self.changepoints)
        self.changepoints_t = sort((self.changepoints .- self.start) ./ self.t_scale)
    else
        self.changepoints_t = [0.0]  # Dummy changepoint
    end
end


function fourier_series(dates::Array{Dates.DateTime}, period::Real, series_order::Int)::Array{Float64,2}
    if series_order < 1
        error("series_order must be >= 1")
    end

    t = Dates.value.(dates) / (10^9 * 3600 * 24)
    x_T = t .* π .* 2
    fourier_components = zeros(length(dates), 2 * series_order)
    for i in 1:series_order
        c = x_T .* i ./ period
        fourier_components[:, 2 * i - 1] = sin.(c)
        fourier_components[:, 2 * i] = cos.(c)
    end
    return fourier_components
end

function make_seasonality_features(dates, period, series_order, prefix)
    features = fourier_series(dates, period, series_order)
    columns = ["$(prefix)_delim_$(i)" for i in 1:size(features, 2)]
    return DataFrame(features, Symbol.(columns))
end


function construct_holiday_dataframe(
    m, dates::Vector{Date}
)
    all_holidays = DataFrame()

    if !isempty(m.holidays)
        all_holidays = copy(m.holidays)
    end

    if m.country_holidays !== nothing
        year_list = unique([x.year for x in dates])
        country_holidays_df = make_holidays_df(
            year_list, m.country_holidays
        )
        all_holidays = vcat(all_holidays, country_holidays_df)
        all_holidays = unique(all_holidays)
    end

    # Drop future holidays not previously seen in training data
    if m.train_holiday_names !== nothing
        # Remove holiday names that didn't show up in fit
        all_holidays = all_holidays[all_holidays[:, :holiday] .∈ Ref(m.train_holiday_names), :]

        # Add holiday names in fit but not in predict with ds as missing
        holidays_to_add = DataFrame(
            holiday = m.train_holiday_names[.!(m.train_holiday_names .∈ Ref(all_holidays.holiday))]
        )
        all_holidays = vcat(all_holidays, holidays_to_add)
        all_holidays = unique(all_holidays)
    end

    return all_holidays
end


function make_holiday_features(
    m, dates::Vector{Date}, holidays::DataFrame
)
    # Holds columns of our future matrix.
    expanded_holidays = DefaultDict(zeros(length(dates)))
    prior_scales = Dict{String, Float64}()

    for row in eachrow(holidays)
        dt = row.ds
        lw = get(row, :lower_window, 0)
        uw = get(row, :upper_window, 0)
        ps = get(row, :prior_scale, m.holidays_prior_scale)

        if ps <= 0
            error("Prior scale must be > 0")
        end

        if haskey(prior_scales, row.holiday) && prior_scales[row.holiday] != ps
            error("Holiday $(row.holiday) does not have consistent prior scale specification.")
        end

        prior_scales[row.holiday] = ps

        for offset in lw:uw
            occurrence = dt + Day(offset)
            loc = findfirst(isequal(occurrence), dates)
            key = string(row.holiday, "_delim_", offset >= 0 ? "+" : "-", abs(offset))

            if loc !== nothing
                expanded_holidays[key][loc] = 1.0
            else
                expanded_holidays[key]  # Access key to generate value
            end
        end
    end

    holiday_features = DataFrame(expanded_holidays)
    sort!(holiday_features, names(holiday_features))
    prior_scale_list = [prior_scales[split(h, "_delim_")[1]] for h in names(holiday_features)]
    holiday_names = keys(prior_scales)

    # Store holiday names used in fit
    if m.train_holiday_names === nothing
        m.train_holiday_names = holiday_names
    end

    return holiday_features, prior_scale_list
end


function add_regressor(
    m, name, 
    prior_scale=nothing, 
    standardize="auto", 
    mode=nothing
)
    if !isnothing(m.history)
        error("Regressors must be added prior to model fitting.")
    end

    validate_column_name(m, name, check_regressors=false)
    if isnothing(prior_scale)
        prior_scale = m.holidays_prior_scale
    end
    if isnothing(mode)
        mode = m.seasonality_mode
    end

    if prior_scale <= 0
        error("Prior scale must be > 0")
    end

    if mode ∉ ["additive", "multiplicative"]
        error("mode must be 'additive' or 'multiplicative'")
    end

    m.extra_regressors[name] = Dict(
        "prior_scale" => prior_scale,
        "standardize" => standardize,
        "mu" => 0.0,
        "std" => 1.0,
        "mode" => mode
    )

    return m
end


function add_seasonality(
    m, name, period, fourier_order; 
    prior_scale=nothing, mode=nothing, condition_name=nothing
)
    if !isnothing(m.history)
        error("Seasonality must be added prior to model fitting.")
    end

    if name ∉ ["daily", "weekly", "yearly"]
        # Allow overwriting built-in seasonalities
        validate_column_name(m, name, check_seasonalities=false)
    end

    if isnothing(prior_scale)
        ps = m.seasonality_prior_scale
    else
        ps = prior_scale
    end

    if ps <= 0
        error("Prior scale must be > 0")
    end

    if fourier_order <= 0
        error("Fourier Order must be > 0")
    end

    if isnothing(mode)
        mode = m.seasonality_mode
    end

    if mode ∉ ["additive", "multiplicative"]
        error("mode must be 'additive' or 'multiplicative'")
    end

    if !isnothing(condition_name)
        validate_column_name(m, condition_name)
    end

    m.seasonalities[name] = Dict(
        "period" => period,
        "fourier_order" => fourier_order,
        "prior_scale" => ps,
        "mode" => mode,
        "condition_name" => condition_name
    )

    return m
end


function add_country_holidays(m, country_name)
    if !isnothing(m.history)
        error("Country holidays must be added prior to model fitting.")
    end

    # Validate names.
    for name in get_holiday_names(country_name)
        # Allow merging with existing holidays
        validate_column_name(m, name, check_holidays=false)
    end

    # Set the holidays.
    if !isnothing(m.country_holidays)
        @warn "Changing country holidays from $(m.country_holidays) to $(country_name)."
    end
    m.country_holidays = country_name

    return m
end


function make_all_seasonality_features(m, df)
    seasonal_features = []
    prior_scales = []
    modes = Dict("additive" => [], "multiplicative" => [])

    # Seasonality features
    for (name, props) in m.seasonalities
        features = make_seasonality_features(m, df.ds, props["period"], props["fourier_order"], name)
        if !isnothing(props["condition_name"])
            features[.!df[:, props["condition_name"]], :] .= 0
        end
        push!(seasonal_features, features)
        append!(prior_scales, fill(props["prior_scale"], size(features, 2)))
        push!(modes[props["mode"]], name)
    end

    # Holiday features
    holidays = construct_holiday_dataframe(m, df.ds)
    if size(holidays, 1) > 0
        features, holiday_priors, holiday_names = make_holiday_features(m, df.ds, holidays)
        push!(seasonal_features, features)
        append!(prior_scales, holiday_priors)
        append!(modes[m.seasonality_mode], holiday_names)
    end

    # Additional regressors
    for (name, props) in m.extra_regressors
        push!(seasonal_features, DataFrame(df[:, name]))
        push!(prior_scales, props["prior_scale"])
        push!(modes[props["mode"]], name)
    end

    # Dummy to prevent empty X
    if isempty(seasonal_features)
        push!(seasonal_features, DataFrame(zeros = zeros(size(df, 1))))
        push!(prior_scales, 1.)
    end

    seasonal_features = hcat(seasonal_features...)
    component_cols, modes = regressor_column_matrix(m, seasonal_features, modes)
    return seasonal_features, prior_scales, component_cols, modes
end


function regressor_column_matrix(m, seasonal_features, modes)
    components = DataFrame(
        col = 1:size(seasonal_features, 2),
        component = [split(x, "_delim_")[1] for x in names(seasonal_features)]
    )

    # Add total for holidays
    if !isnothing(m.train_holiday_names)
        components = add_group_component(m, components, "holidays", unique(m.train_holiday_names))
    end

    # Add totals additive and multiplicative components, and regressors
    for mode in ["additive", "multiplicative"]
        components = add_group_component(m, components, mode * "_terms", modes[mode])

        regressors_by_mode = [
            r for (r, props) in m.extra_regressors
            if props["mode"] == mode
        ]
        components = add_group_component(m, components, "extra_regressors_" * mode, regressors_by_mode)

        # Add combination components to modes
        push!(modes[mode], mode * "_terms")
        push!(modes[mode], "extra_regressors_" * mode)
    end

    # After all of the additive/multiplicative groups have been added,
    push!(modes[m.seasonality_mode], "holidays")

    # Convert to a binary matrix
    component_cols = unstack(components, :col, :component, :component, fill = 0)
    component_cols = component_cols[sort(names(component_cols))]

    # Add columns for additive and multiplicative terms, if missing
    for name in ["additive_terms", "multiplicative_terms"]
        if name ∉ names(component_cols)
            component_cols[!, name] .= 0
        end
    end

    # Remove the placeholder
    select!(component_cols, Not(:zeros))

    # Validation
    if maximum(component_cols.additive_terms .+ component_cols.multiplicative_terms) > 1
        error("A bug occurred in seasonal components.")
    end

    # Compare to the training, if set.
    if !isnothing(m.train_component_cols)
        component_cols = component_cols[:, names(m.train_component_cols)]
        if !isequal(component_cols, m.train_component_cols)
            error("A bug occurred in constructing regressors.")
        end
    end

    return component_cols, modes
end


function add_group_component(m, components, name, group)
    new_comp = components[in.(components.component, Ref(Set(group))), :]
    group_cols = unique(new_comp.col)
    if !isempty(group_cols)
        new_comp = DataFrame(col = group_cols, component = name)
        components = vcat(components, new_comp)
    end
    return components
end


function parse_seasonality_args(m, name, arg, auto_disable, default_order)
    if arg == "auto"
        fourier_order = 0
        if haskey(m.seasonalities, name)
            @info "Found custom seasonality named $name, disabling built-in $name seasonality."
        elseif auto_disable
            @info "Disabling $name seasonality. Run prophet with $name_seasonality=True to override this."
        else
            fourier_order = default_order
        end
    elseif arg == true
        fourier_order = default_order
    elseif arg == false
        fourier_order = 0
    else
        fourier_order = Int(arg)
    end
    return fourier_order
end


function set_auto_seasonalities(m)
    first = minimum(m.history[:, :ds])
    last = maximum(m.history[:, :ds])
    dt = diff(m.history[:, :ds])
    min_dt = minimum(dt[dt .!= Dates.Day(0)])

    # Yearly seasonality
    yearly_disable = last - first < Dates.Day(730)
    fourier_order = parse_seasonality_args(m, "yearly", m.yearly_seasonality, yearly_disable, 10)
    if fourier_order > 0
        m.seasonalities["yearly"] = Dict(
            "period" => 365.25,
            "fourier_order" => fourier_order,
            "prior_scale" => m.seasonality_prior_scale,
            "mode" => m.seasonality_mode,
            "condition_name" => nothing
        )
    end

    # Weekly seasonality
    weekly_disable = (last - first < Dates.Day(7 * 2)) || (min_dt >= Dates.Day(7))
    fourier_order = parse_seasonality_args(m, "weekly", m.weekly_seasonality, weekly_disable, 3)
    if fourier_order > 0
        m.seasonalities["weekly"] = Dict(
            "period" => 7,
            "fourier_order" => fourier_order,
            "prior_scale" => m.seasonality_prior_scale,
            "mode" => m.seasonality_mode,
            "condition_name" => nothing
        )
    end

    # Daily seasonality
    daily_disable = (last - first < Dates.Day(2)) || (min_dt >= Dates.Day(1))
    fourier_order = parse_seasonality_args(m, "daily", m.daily_seasonality, daily_disable, 4)
    if fourier_order > 0
        m.seasonalities["daily"] = Dict(
            "period" => 1,
            "fourier_order" => fourier_order,
            "prior_scale" => m.seasonality_prior_scale,
            "mode" => m.seasonality_mode,
            "condition_name" => nothing
        )
    end
end


function linear_growth_init(df)
    i0, i1 = argmin(df[:, :ds]), argmax(df[:, :ds])
    T = df[i1, :t] - df[i0, :t]
    k = (df[i1, :y_scaled] - df[i0, :y_scaled]) / T
    m = df[i0, :y_scaled] - k * df[i0, :t]
    return k, m
end

function logistic_growth_init(df)
    i0, i1 = argmin(df[:, :ds]), argmax(df[:, :ds])
    T = df[i1, :t] - df[i0, :t]

    C0 = df[i0, :cap_scaled]
    C1 = df[i1, :cap_scaled]
    y0 = max(0.01 * C0, min(0.99 * C0, df[i0, :y_scaled]))
    y1 = max(0.01 * C1, min(0.99 * C1, df[i1, :y_scaled]))

    r0 = C0 / y0
    r1 = C1 / y1

    if abs(r0 - r1) <= 0.01
        r0 = 1.05 * r0
    end

    L0 = log(r0 - 1)
    L1 = log(r1 - 1)

    m = L0 * T / (L0 - L1)
    k = (L0 - L1) / T
    return k, m
end

function flat_growth_init(df)
    k = 0
    m = mean(df[:, :y_scaled])
    return k, m
end


function fit(prophet, df; kwargs...)
    if !isnothing(prophet.history)
        error("Prophet object can only be fit once. Instantiate a new object.")
    end
    if !("ds" in names(df)) || !("y" in names(df))
        error("Dataframe must have columns ds and y with the dates and values respectively.")
    end
    history = df[.!ismissing.(df.y), :]
    if nrow(history) < 2
        error("Dataframe has less than 2 non-NaN rows.")
    end
    prophet.history_dates = sort(unique(df.ds))

    history = setup_dataframe(prophet, history, initialize_scales=true)
    prophet.history = history
    set_auto_seasonalities(prophet)
    seasonal_features, prior_scales, component_cols, modes = make_all_seasonality_features(prophet, history)
    prophet.train_component_cols = component_cols
    prophet.component_modes = modes
    prophet.fit_kwargs = deepcopy(kwargs)

    set_changepoints(prophet)

    trend_indicator = Dict("linear" => 0, "logistic" => 1, "flat" => 2)

    dat = Dict(
        "T" => nrow(history),
        "K" => size(seasonal_features, 2),
        "S" => length(prophet.changepoints_t),
        "y" => history.y_scaled,
        "t" => history.t,
        "t_change" => prophet.changepoints_t,
        "X" => seasonal_features,
        "sigmas" => prior_scales,
        "tau" => prophet.changepoint_prior_scale,
        "trend_indicator" => trend_indicator[prophet.growth],
        "s_a" => component_cols["additive_terms"],
        "s_m" => component_cols["multiplicative_terms"]
    )

    if prophet.growth == "linear"
        dat["cap"] = zeros(nrow(history))
        kinit = linear_growth_init(history)
    elseif prophet.growth == "flat"
        dat["cap"] = zeros(nrow(history))
        kinit = flat_growth_init(history)
    else
        dat["cap"] = history.cap_scaled
        kinit = logistic_growth_init(history)
    end

    stan_init = Dict(
        "k" => kinit[1],
        "m" => kinit[2],
        "delta" => zeros(length(prophet.changepoints_t)),
        "beta" => zeros(size(seasonal_features, 2)),
        "sigma_obs" => 1.0
    )

    if minimum(history.y) == maximum(history.y) && (prophet.growth == "linear" || prophet.growth == "flat")
        prophet.params = stan_init
        prophet.params["sigma_obs"] = 1e-9
        for par in keys(prophet.params)
            prophet.params[par] = [prophet.params[par]]
        end
    elseif prophet.mcmc_samples > 0
        # Replace with the appropriate Stan sampling function call
        prophet.params = stan_sampling(stan_init, dat, prophet.mcmc_samples, kwargs...)
    else
        # Replace with the appropriate Stan optimization function call
        prophet.params = stan_optimization(stan_init, dat, kwargs...)
    end

    prophet.stan_fit = prophet.stan_backend.stan_fit

    if isempty(prophet.changepoints)
        prophet.params["k"] = prophet.params["k"] .+ reshape(prophet.params["delta"], -1)
        prophet.params["delta"] = reshape(zeros(size(prophet.params["delta"])), (-1, 1))
    end

    return prophet
end


function predict(prophet, df=nothing, vectorized=true)
    if prophet.history === nothing
        error("Model has not been fit.")
    end

    if df === nothing
        df = deepcopy(prophet.history)
    else
        if nrow(df) == 0
            error("Dataframe has no rows.")
        end
        df = setup_dataframe(prophet, deepcopy(df))
    end

    df[!, "trend"] = predict_trend(prophet, df)
    seasonal_components = predict_seasonal_components(prophet, df)
    if prophet.uncertainty_samples
        intervals = predict_uncertainty(prophet, df, vectorized)
    else
        intervals = nothing
    end

    cols = ["ds", "trend"]
    if "cap" in names(df)
        push!(cols, "cap")
    end
    if prophet.logistic_floor
        push!(cols, "floor")
    end

    df2 = hcat(df[:, cols], intervals, seasonal_components)
    df2[!, "yhat"] = df2[!, "trend"] .* (1 .+ df2[!, "multiplicative_terms"]) .+ df2[!, "additive_terms"]
    return df2
end

function predict_seasonal_components(prophet, df)
    seasonal_features, _, component_cols, _ = make_all_seasonality_features(prophet, df)
    if prophet.uncertainty_samples
        lower_p = 100 * (1.0 - prophet.interval_width) / 2
        upper_p = 100 * (1.0 + prophet.interval_width) / 2
    end

    X = Matrix(seasonal_features)
    data = Dict{Symbol, Vector}()
    for component in names(component_cols)
        beta_c = prophet.params["beta"] .* component_cols[!, component]

        comp = X * transpose(beta_c)
        if component in prophet.component_modes["additive"]
            comp .*= prophet.y_scale
        end
        data[Symbol(component)] = vec(mean(comp, dims=1))
        if prophet.uncertainty_samples
            data[Symbol(component * "_lower")] = vec(percentile(comp, lower_p, dims=1))
            data[Symbol(component * "_upper")] = vec(percentile(comp, upper_p, dims=1))
        end
    end
    return DataFrame(data)
end


function piecewise_linear(t, deltas, k, m, changepoint_ts)
    deltas_t = (changepoint_ts .<= t) .* deltas'
    k_t = sum(deltas_t, dims=2) .+ k
    m_t = sum(deltas_t .* -changepoint_ts, dims=2) .+ m
    return k_t .* t .+ m_t
end

function piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
    k_cum = vcat(k, cumsum(deltas) .+ k)
    gammas = zeros(length(changepoint_ts))
    for i in 1:length(changepoint_ts)
        t_s = changepoint_ts[i]
        gammas[i] = (t_s - m - sum(gammas)) * (1 - k_cum[i] / k_cum[i + 1])
    end
    k_t = k * ones(size(t))
    m_t = m * ones(size(t))
    for s in 1:length(changepoint_ts)
        t_s = changepoint_ts[s]
        indx = t .>= t_s
        k_t[indx] .+= deltas[s]
        m_t[indx] .+= gammas[s]
    end
    return cap ./ (1 .+ exp.(-k_t .* (t .- m_t)))
end

function flat_trend(t, m)
    m_t = m * ones(size(t))
    return m_t
end

function predict_trend(prophet, df)
    k = nanmean(prophet.params["k"])
    m = nanmean(prophet.params["m"])
    deltas = nanmean(prophet.params["delta"], dims=1)

    t = df[!, "t"]
    if prophet.growth == "linear"
        trend = piecewise_linear(t, deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "logistic"
        cap = df[!, "cap_scaled"]
        trend = piecewise_logistic(t, cap, deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "flat"
        trend = flat_trend(t, m)
    end

    return trend .* prophet.y_scale .+ df[!, "floor"]
end


using DataFrames
using Random

function sample_predictive_trend_vectorized(prophet, df, n_samples, iteration = 1)
    deltas = prophet.params["delta"][iteration, :]
    m = prophet.params["m"][iteration]
    k = prophet.params["k"][iteration]

    if prophet.growth == "linear"
        expected = piecewise_linear(df[!, "t"], deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "logistic"
        expected = piecewise_logistic(df[!, "t"], df[!, "cap_scaled"], deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "flat"
        expected = flat_trend(df[!, "t"], m)
    else
        throw(NotImplementedError())
    end

    uncertainty = _sample_uncertainty(prophet, df, n_samples, iteration)
    return (
        (repeat(expected', n_samples, 1) .+ uncertainty) .* prophet.y_scale .+
        repeat(df[!, "floor"]', n_samples, 1)
    )
end

function sample_model_vectorized(prophet, df, seasonal_features, iteration, s_a, s_m, n_samples)
    beta = prophet.params["beta"][iteration, :]
    Xb_a = (seasonal_features * beta .* s_a') .* prophet.y_scale
    Xb_m = seasonal_features * beta .* s_m'
    
    trends = sample_predictive_trend_vectorized(prophet, df, n_samples, iteration)
    sigma = prophet.params["sigma_obs"][iteration]
    noise_terms = randn(size(trends)) .* sigma .* prophet.y_scale

    simulations = [
        Dict(
            "yhat" => trend .* (1 .+ Xb_m) .+ Xb_a .+ noise,
            "trend" => trend
        ) for (trend, noise) in zip(eachrow(trends), eachrow(noise_terms))
    ]

    return simulations
end

function sample_posterior_predictive(prophet, df, vectorized)
    n_iterations = size(prophet.params["k"])[1]
    samp_per_iter = max(1, Int(ceil(
        prophet.uncertainty_samples / n_iterations
    )))

    seasonal_features, _, component_cols, _ = make_all_seasonality_features(prophet, df)
    sim_values = Dict("yhat" => [], "trend" => [])

    for i in 1:n_iterations
        if vectorized
            sims = sample_model_vectorized(
                prophet, df, seasonal_features, i, component_cols[:, "additive_terms"],
                component_cols[:, "multiplicative_terms"], samp_per_iter
            )
        else
            sims = [
                sample_model(
                    prophet, df, seasonal_features, i, component_cols[:, "additive_terms"],
                    component_cols[:, "multiplicative_terms"]
                ) for _ in 1:samp_per_iter
            ]
        end
        for key in keys(sim_values)
            for sim in sims
                push!(sim_values[key], sim[key])
            end
        end
    end
    for k in keys(sim_values)
        sim_values[k] = hcat(sim_values[k]...)
    end
    return sim_values
end


using DataFrames
using Statistics

function predict_uncertainty(prophet, df, vectorized)
    sim_values = sample_posterior_predictive(prophet, df, vectorized)

    lower_p = 100 * (1.0 - prophet.interval_width) / 2
    upper_p = 100 * (1.0 + prophet.interval_width) / 2

    series = Dict()
    for key in ["yhat", "trend"]
        series["$(key)_lower"] = percentile(sim_values[key], lower_p, dims=2)
        series["$(key)_upper"] = percentile(sim_values[key], upper_p, dims=2)
    end

    return DataFrame(series)
end

function sample_model(prophet, df, seasonal_features, iteration, s_a, s_m)
    trend = sample_predictive_trend(prophet, df, iteration)

    beta = prophet.params["beta"][iteration, :]
    Xb_a = (seasonal_features * beta .* s_a') .* prophet.y_scale
    Xb_m = seasonal_features * beta .* s_m'

    sigma = prophet.params["sigma_obs"][iteration]
    noise = randn(nrow(df)) .* sigma .* prophet.y_scale

    return Dict("yhat" => trend .* (1 .+ Xb_m) .+ Xb_a .+ noise, "trend" => trend)
end

function sample_predictive_trend(prophet, df, iteration)
    k = prophet.params["k"][iteration]
    m = prophet.params["m"][iteration]
    deltas = prophet.params["delta"][iteration, :]

    t = df[!, "t"]
    T = maximum(t)

    if T > 1
        S = length(prophet.changepoints_t)
        n_changes = rand(Poisson(S * (T - 1)))
    else
        n_changes = 0
    end

    if n_changes > 0
        changepoint_ts_new = 1 .+ rand(n_changes) .* (T - 1)
        sort!(changepoint_ts_new)
    else
        changepoint_ts_new = Float64[]
    end

    lambda_ = mean(abs.(deltas)) + 1e-8
    deltas_new = rand(Laplace(0, lambda_), n_changes)

    changepoint_ts = vcat(prophet.changepoints_t, changepoint_ts_new)
    deltas = vcat(deltas, deltas_new)

    if prophet.growth == "linear"
        trend = piecewise_linear(t, deltas, k, m, changepoint_ts)
    elseif prophet.growth == "logistic"
        cap = df[!, "cap_scaled"]
        trend = piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
    elseif prophet.growth == "flat"
        trend = flat_trend(t, m)
    end

    return trend .* prophet.y_scale .+ df[!, "floor"]
end

function sample_predictive_trend_vectorized(prophet, df, n_samples, iteration=1)
    deltas = prophet.params["delta"][iteration, :]
    m = prophet.params["m"][iteration]
    k = prophet.params["k"][iteration]
    t = df[!, "t"]

    if prophet.growth == "linear"
        expected = piecewise_linear(t, deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "logistic"
        expected = piecewise_logistic(t, df[!, "cap_scaled"], deltas, k, m, prophet.changepoints_t)
    elseif prophet.growth == "flat"
        expected = flat_trend(t, m)
    else
        error("NotImplementedError")
    end

    uncertainty = _sample_uncertainty(prophet, df, n_samples, iteration)

    return (
        (repeat(expected', n_samples, 1) .+ uncertainty) .* prophet.y_scale .+
        repeat(df[!, "floor"]', n_samples, 1)
    )
end

function _sample_uncertainty(prophet, df, n_samples, iteration=1)
    tmax = maximum(df[!, "t"])

    if tmax <= 1
        uncertainties = zeros(n_samples, nrow(df))
    else
        future_df = df[df[!, "t"] .> 1, :]
        n_length = nrow(future_df)

        if n_length > 1
            single_diff = mean(diff(future_df[!, "t"]))
        else
            single_diff = mean(diff(prophet.history[!, "t"]))
        end

        change_likelihood = length(prophet.changepoints_t) * single_diff
        deltas = prophet.params["delta"][iteration, :]
        m = prophet.params["m"][iteration]
        k = prophet.params["k"][iteration]
        mean_delta = mean(abs.(deltas)) + 1e-8

        if prophet.growth == "linear"
            mat = _make_trend_shift_matrix(prophet, mean_delta, change_likelihood, n_length, n_samples=n_samples)
            uncertainties = cumsum(mat, dims=1)
            uncertainties = cumsum(uncertainties, dims=1)
            uncertainties .*= single_diff
        elseif prophet.growth == "logistic"
            mat = _make_trend_shift_matrix(prophet, mean_delta, change_likelihood, n_length, n_samples=n_samples)
            uncertainties = _logistic_uncertainty(
                prophet, mat, deltas, k, m, future_df[!, "cap_scaled"],
                future_df[!, "t"], n_length, single_diff
            )
        elseif prophet.growth == "flat"
            uncertainties = zeros(n_samples, n_length)
        else
            error("NotImplementedError")
        end

        if minimum(df[!, "t"]) <= 1
            past_uncertainty = zeros(n_samples, sum(df[!, "t"] .<= 1))
            uncertainties = hcat(past_uncertainty, uncertainties)
        end
    end

    return uncertainties
end


function _make_trend_shift_matrix(mean_delta, likelihood, future_length, n_samples)
    bool_slope_change = rand(n_samples, future_length) .< likelihood
    shift_values = rand(Laplace(0, mean_delta), size(bool_slope_change))
    mat = shift_values .* bool_slope_change
    n_mat = hcat(zeros(size(mat, 1), 1), mat)[:, 1:end-1]
    mat = (n_mat .+ mat) ./ 2
    return mat
end

function _make_historical_mat_time(deltas, changepoints_t, t_time, n_row=1, single_diff=nothing)
    if single_diff === nothing
        single_diff = mean(diff(t_time))
    end
    prev_time = range(0, 1 + single_diff, step=single_diff)
    idxs = [findfirst(>(changepoint), prev_time) for changepoint in changepoints_t]
    prev_deltas = zeros(length(prev_time))
    prev_deltas[idxs] .= deltas
    prev_deltas = repeat(prev_deltas', n_row, 1)
    return prev_deltas, prev_time
end

function _logistic_uncertainty(prophet, mat, deltas, k, m, cap, t_time, n_length, single_diff=nothing)
    function ffill(arr)
        mask = arr .== 0
        idx = mask ?|> x -> x ? 0 : 1
        idx = cummax(idx, dims=2)
        return arr[:, idx]
    end

    historical_mat, historical_time = _make_historical_mat_time(deltas, prophet.changepoints_t, t_time, size(mat, 1), single_diff)
    mat = hcat(historical_mat, mat)
    full_t_time = vcat(historical_time, t_time)

    k_cum = hcat(fill(k, (size(mat, 1), 1)), mat .|> x -> x ? cumsum(mat, dims=2) + k : 0)
    k_cum_b = ffill(k_cum)
    gammas = zeros(size(mat))
    for i in 1:size(mat, 2)
        x = full_t_time[i] - m - sum(gammas[:, 1:i-1], dims=2)
        ks = 1 .- k_cum_b[:, i] ./ k_cum_b[:, i + 1]
        gammas[:, i] = x .* ks
    end

    k_t = (cumsum(mat, dims=2) .+ k)[:, end - n_length + 1:end]
    m_t = (cumsum(gammas, dims=2) .+ m)[:, end - n_length + 1:end]
    sample_trends = cap ./ (1 .+ exp.(-k_t .* (t_time .- m_t)))
    return sample_trends .- mean(sample_trends, dims=1)
end


function predictive_samples(prophet, df::DataFrame, vectorized::Bool=true)
    df = prophet.setup_dataframe(copy(df))
    return prophet.sample_posterior_predictive(df, vectorized)
end

function percentile(prophet, a; args...)
    fn = any(isnan, a) ? nanquantile : quantile
    return fn(a, args...; dims=1)
end

function make_future_dataframe(prophet, periods; freq=Day(1), include_history=true)
    if prophet.history_dates === nothing
        error("Model has not been fit.")
    end

    if freq === nothing
        freq = DateFrequency.infer_freq(last(prophet.history_dates, 5))
        if freq === nothing
            error("Unable to infer `freq`")
        end
    end

    last_date = maximum(prophet.history_dates)
    dates = collect(DateRange(last_date, periods + 1, step=freq))
    dates = dates[dates .> last_date]
    dates = dates[1:periods]

    if include_history
        dates = vcat(prophet.history_dates, dates)
    end

    return DataFrame(ds=dates)
end


function plot_gadfly(prophet, fcst; uncertainty=true, plot_cap=true,
                     xlabel="ds", ylabel="y", include_legend=false)
    # Implement your plot_forecast_gadfly function here
    return plot_forecast_gadfly(
        prophet, fcst, uncertainty=uncertainty,
        plot_cap=plot_cap, xlabel=xlabel, ylabel=ylabel,
        include_legend=include_legend
    )
end

function plot_components_gadfly(prophet, fcst; uncertainty=true, plot_cap=true,
                                weekly_start=0, yearly_start=0)
    # Implement your plot_forecast_components_gadfly function here
    return plot_forecast_components_gadfly(
        prophet, fcst, uncertainty=uncertainty, plot_cap=plot_cap,
        weekly_start=weekly_start, yearly_start=yearly_start
    )
end


function plot_plotly(prophet, fcst; uncertainty=true, plot_cap=true,
                     xlabel="ds", ylabel="y", include_legend=false)
    # Implement your plot_forecast_plotly function here
    return plot_forecast_plotly(
        prophet, fcst, uncertainty=uncertainty,
        plot_cap=plot_cap, xlabel=xlabel, ylabel=ylabel,
        include_legend=include_legend
    )
end

function plot_components_plotly(prophet, fcst; uncertainty=true, plot_cap=true,
                                weekly_start=0, yearly_start=0)
    # Implement your plot_forecast_components_plotly function here
    return plot_forecast_components_plotly(
        prophet, fcst, uncertainty=uncertainty, plot_cap=plot_cap,
        weekly_start=weekly_start, yearly_start=yearly_start
    )
end


