using DataFrames
using Dates
using Stan


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

using DataFrames
using Dates

function validate_inputs(self)
    if !(self.growth in ["linear", "logistic", "flat"])
        error('Parameter "growth" should be "linear", "logistic" or "flat".')
    end
    if !(isa(self.changepoint_range, Real) && 0 <= self.changepoint_range <= 1)
        error(Parameter "changepoint_range" must be in [0, 1]')
    end
    if !isnothing(self.holidays)
        if !(isa(self.holidays, DataFrame) && "ds" in names(self.holidays) && "holiday" in names(self.holidays))
            error('holidays must be a DataFrame with "ds" and "holiday" columns.')
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

