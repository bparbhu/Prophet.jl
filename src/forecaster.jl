using CSV
using DataFrames
using Dates
using LinearAlgebra
using Statistics
using Turing

mutable struct ProphetModel
    growth::String
    changepoints::Union{Nothing,Vector{Date}}
    specified_changepoints::Bool
    model_backend::Symbol
    n_changepoints::Int
    changepoint_range::Float64
    yearly_seasonality::Union{String,Bool,Int}
    weekly_seasonality::Union{String,Bool,Int}
    daily_seasonality::Union{String,Bool,Int}
    country_holidays::Union{Nothing,String}
    holidays::Union{Nothing,DataFrame}
    seasonality_mode::String
    holidays_mode::String
    seasonality_prior_scale::Float64
    holidays_prior_scale::Float64
    changepoint_prior_scale::Float64
    mcmc_samples::Int
    interval_width::Float64
    uncertainty_samples::Int
    scaling::String
    logistic_floor::Bool
    history::Union{Nothing,DataFrame}
    history_dates::Union{Nothing,Vector{Date}}
    start::Union{Nothing,Date}
    y_min::Union{Nothing,Float64}
    y_scale::Union{Nothing,Float64}
    t_scale::Union{Nothing,Period}
    params::Dict{String,Any}
    fit_backend::Union{Nothing,Symbol}
    fit_engine::Union{Nothing,Symbol}
    fit_result::Any
    fit_kwargs::Dict{String,Any}
    backend_data::Dict{String,Any}
    seasonalities::Dict{String,Dict{String,Any}}
    extra_regressors::Dict{String,Dict{String,Any}}
    train_component_cols::Union{Nothing,DataFrame}
    component_modes::Union{Nothing,Dict{String,Vector{String}}}
    train_holiday_names::Union{Nothing,Vector{String}}
    changepoints_t::Union{Nothing,Vector{Float64}}
end

function ProphetModel(;
    growth::AbstractString="linear",
    changepoints=nothing,
    model_backend::Union{Symbol,AbstractString}=:stan,
    stan_backend=nothing,
    n_changepoints::Integer=25,
    changepoint_range::Real=0.8,
    yearly_seasonality::Union{String,Bool,Integer}="auto",
    weekly_seasonality::Union{String,Bool,Integer}="auto",
    daily_seasonality::Union{String,Bool,Integer}="auto",
    holidays::Union{Nothing,DataFrame}=nothing,
    country_holidays::Union{Nothing,AbstractString}=nothing,
    seasonality_mode::AbstractString="additive",
    holidays_mode::Union{Nothing,AbstractString}=nothing,
    seasonality_prior_scale::Real=10.0,
    holidays_prior_scale::Real=10.0,
    changepoint_prior_scale::Real=0.05,
    mcmc_samples::Integer=0,
    interval_width::Real=0.8,
    uncertainty_samples::Integer=1000,
    scaling::AbstractString="absmax",
)
    growth in ("linear", "logistic", "flat") ||
        error("Parameter \"growth\" should be \"linear\", \"logistic\" or \"flat\".")
    resolved_backend = _normalize_model_backend(model_backend, stan_backend)
    0 <= changepoint_range <= 1 || error("changepoint_range must be in [0, 1].")
    seasonality_mode in ("additive", "multiplicative") ||
        error("seasonality_mode must be \"additive\" or \"multiplicative\".")
    resolved_holidays_mode = isnothing(holidays_mode) ? String(seasonality_mode) : String(holidays_mode)
    resolved_holidays_mode in ("additive", "multiplicative") ||
        error("holidays_mode must be \"additive\" or \"multiplicative\".")
    scaling in ("absmax", "minmax") || error("scaling must be one of \"absmax\" or \"minmax\".")
    resolved_changepoints = isnothing(changepoints) ? nothing : Date.(changepoints)
    return ProphetModel(
        String(growth),
        resolved_changepoints,
        !isnothing(resolved_changepoints),
        resolved_backend,
        Int(n_changepoints),
        Float64(changepoint_range),
        yearly_seasonality,
        weekly_seasonality,
        daily_seasonality,
        isnothing(country_holidays) ? nothing : uppercase(String(country_holidays)),
        holidays,
        String(seasonality_mode),
        resolved_holidays_mode,
        Float64(seasonality_prior_scale),
        Float64(holidays_prior_scale),
        Float64(changepoint_prior_scale),
        Int(mcmc_samples),
        Float64(interval_width),
        Int(uncertainty_samples),
        String(scaling),
        false,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        Dict{String,Any}(),
        nothing,
        nothing,
        nothing,
        Dict{String,Any}(),
        Dict{String,Any}(),
        Dict{String,Dict{String,Any}}(),
        Dict{String,Dict{String,Any}}(),
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function _normalize_model_backend(model_backend, stan_backend)
    backend = Symbol(model_backend)
    if stan_backend !== nothing
        backend = :stan
    end
    backend in (:stan, :turing, :neural_turing, :flux_turing) ||
        error("model_backend must be :stan, :turing, or :neural_turing.")
    backend == :flux_turing && return :neural_turing
    return backend
end

function validate_inputs(m::ProphetModel)
    m.growth in ("linear", "logistic", "flat") ||
        error("Parameter \"growth\" should be \"linear\", \"logistic\" or \"flat\".")
    0 <= m.changepoint_range <= 1 || error("changepoint_range must be in [0, 1].")
    m.seasonality_mode in ("additive", "multiplicative") ||
        error("seasonality_mode must be \"additive\" or \"multiplicative\".")
    m.holidays_mode in ("additive", "multiplicative") ||
        error("holidays_mode must be \"additive\" or \"multiplicative\".")
    m.mcmc_samples >= 0 || error("mcmc_samples must be greater than or equal to 0.")
    0 <= m.interval_width <= 1 || error("interval_width must be between 0 and 1.")
    m.uncertainty_samples >= 0 || error("uncertainty_samples must be greater than or equal to 0.")
    if m.holidays !== nothing
        all(in.(["ds", "holiday"], Ref(names(m.holidays)))) ||
            error("holidays must be a DataFrame with ds and holiday columns.")
    end
    return true
end

function set_model_backend!(m::ProphetModel, backend)
    m.history !== nothing && error("Model backend must be selected before fitting.")
    m.model_backend = _normalize_model_backend(backend, nothing)
    return m
end

function validate_column_name(
    m::ProphetModel,
    name::AbstractString;
    check_holidays::Bool=true,
    check_seasonalities::Bool=true,
    check_regressors::Bool=true,
)
    occursin("_delim_", name) && error("Name cannot contain \"_delim_\".")
    reserved = [
        "trend", "additive_terms", "daily", "weekly", "yearly", "holidays", "zeros",
        "extra_regressors_additive", "extra_regressors_multiplicative",
        "multiplicative_terms", "yhat", "ds", "y", "cap", "floor", "y_scaled", "cap_scaled",
    ]
    append!(reserved, [n * "_lower" for n in reserved])
    append!(reserved, [n * "_upper" for n in reserved])
    String(name) in reserved && error("Name \"$(name)\" is reserved.")
    check_holidays && m.holidays !== nothing && String(name) in String.(unique(m.holidays.holiday)) &&
        error("Name \"$(name)\" already used for a holiday.")
    check_holidays && m.country_holidays !== nothing &&
        String(name) in get_holiday_names(m.country_holidays) &&
        error("Name \"$(name)\" is a holiday name in $(m.country_holidays).")
    check_seasonalities && haskey(m.seasonalities, String(name)) &&
        error("Name \"$(name)\" already used for a seasonality.")
    check_regressors && haskey(m.extra_regressors, String(name)) &&
        error("Name \"$(name)\" already used for an added regressor.")
    return true
end

set_model_backend(m::ProphetModel, backend) = set_model_backend!(m, backend)

function model_backend(m::ProphetModel)
    return m.model_backend
end

function fit_backend(m::ProphetModel)
    return m.fit_backend
end

function fit_engine(m::ProphetModel)
    return m.fit_engine
end

function _seasonality_design_matrix(m::ProphetModel, history::DataFrame)
    feature_frames = DataFrame[]
    prior_scales = Float64[]
    modes = String[]

    for (name, props) in sort(collect(m.seasonalities); by=first)
        frame = make_seasonality_features(
            history.ds,
            props["period"],
            props["fourier_order"],
            name,
        )
        condition_name = props["condition_name"]
        if condition_name !== nothing
            condition = Bool.(history[!, Symbol(condition_name)])
            frame[.!condition, :] .= 0.0
        end
        push!(feature_frames, frame)
        append!(prior_scales, fill(Float64(props["prior_scale"]), ncol(frame)))
        append!(modes, fill(String(props["mode"]), ncol(frame)))
    end

    holidays = construct_holiday_dataframe(m, history.ds)
    if nrow(holidays) > 0
        holiday_features, holiday_priors, _ = make_holiday_features(m, history.ds, holidays)
        if ncol(holiday_features) > 0
            push!(feature_frames, holiday_features)
            append!(prior_scales, Float64.(holiday_priors))
            append!(modes, fill(m.holidays_mode, ncol(holiday_features)))
        end
    end

    for (name, props) in sort(collect(m.extra_regressors); by=first)
        column = Symbol(name)
        push!(feature_frames, DataFrame(column => Float64.(history[!, column])))
        push!(prior_scales, Float64(props["prior_scale"]))
        push!(modes, String(props["mode"]))
    end

    if isempty(feature_frames)
        return zeros(nrow(history), 1), [1.0], [0.0], [0.0], String[]
    end

    features = hcat(feature_frames...; makeunique=true)
    X = Matrix{Float64}(features)
    s_a = Float64.(modes .== "additive")
    s_m = Float64.(modes .== "multiplicative")
    return X, prior_scales, s_a, s_m, names(features)
end

function _component_name(feature_name::AbstractString)
    return first(split(String(feature_name), "_delim_"))
end

function add_group_component(components::DataFrame, name::AbstractString, group)
    group_set = Set(String.(group))
    rows = components[in.(components.component, Ref(group_set)), :]
    nrow(rows) == 0 && return components
    return vcat(components, DataFrame(col=unique(rows.col), component=fill(String(name), length(unique(rows.col)))))
end

function regressor_column_matrix(m::ProphetModel, feature_names, modes::Dict{String,Vector{String}})
    components = DataFrame(
        col=collect(1:length(feature_names)),
        component=[_component_name(name) for name in feature_names],
    )
    if m.train_holiday_names !== nothing
        components = add_group_component(components, "holidays", m.train_holiday_names)
    end
    for mode in ("additive", "multiplicative")
        components = add_group_component(components, mode * "_terms", modes[mode])
        regressors_by_mode = [
            name for (name, props) in m.extra_regressors if String(props["mode"]) == mode
        ]
        components = add_group_component(components, "extra_regressors_" * mode, regressors_by_mode)
        push!(modes[mode], mode * "_terms")
        push!(modes[mode], "extra_regressors_" * mode)
    end
    push!(modes[m.holidays_mode], "holidays")

    component_names = sort(unique(components.component))
    out = DataFrame(col=collect(1:length(feature_names)))
    for component in component_names
        out[!, Symbol(component)] = [
            any((components.col .== col) .& (components.component .== component)) ? 1.0 : 0.0
            for col in out.col
        ]
    end
    for name in ("additive_terms", "multiplicative_terms")
        if !(name in names(out))
            out[!, Symbol(name)] = zeros(nrow(out))
        end
    end
    select!(out, Not(:col))
    return out, modes
end

function make_all_seasonality_features(m::ProphetModel, df::DataFrame)
    feature_frames = DataFrame[]
    prior_scales = Float64[]
    modes = Dict("additive" => String[], "multiplicative" => String[])

    for (name, props) in sort(collect(m.seasonalities); by=first)
        frame = make_seasonality_features(df.ds, props["period"], props["fourier_order"], name)
        condition_name = props["condition_name"]
        if condition_name !== nothing
            condition = Bool.(df[!, Symbol(condition_name)])
            frame[.!condition, :] .= 0.0
        end
        push!(feature_frames, frame)
        append!(prior_scales, fill(Float64(props["prior_scale"]), ncol(frame)))
        push!(modes[String(props["mode"])], String(name))
    end

    holidays = construct_holiday_dataframe(m, df.ds)
    if nrow(holidays) > 0
        holiday_features, holiday_priors, holiday_names = make_holiday_features(m, df.ds, holidays)
        if ncol(holiday_features) > 0
            push!(feature_frames, holiday_features)
            append!(prior_scales, Float64.(holiday_priors))
            append!(modes[m.holidays_mode], String.(holiday_names))
        end
    end

    for (name, props) in sort(collect(m.extra_regressors); by=first)
        push!(feature_frames, DataFrame(Symbol(name) => Float64.(df[!, Symbol(name)])))
        push!(prior_scales, Float64(props["prior_scale"]))
        push!(modes[String(props["mode"])], String(name))
    end

    if isempty(feature_frames)
        seasonal_features = DataFrame(zeros=zeros(nrow(df)))
        push!(prior_scales, 1.0)
    else
        seasonal_features = hcat(feature_frames...; makeunique=true)
    end
    component_cols, component_modes = regressor_column_matrix(m, names(seasonal_features), modes)
    return seasonal_features, prior_scales, component_cols, component_modes
end

function _parse_seasonality_args(m::ProphetModel, name::String, arg, auto_disable::Bool, default_order::Int)
    if arg == "auto"
        haskey(m.seasonalities, name) && return 0
        return auto_disable ? 0 : default_order
    elseif arg === true
        return default_order
    elseif arg === false
        return 0
    else
        return Int(arg)
    end
end

parse_seasonality_args(m::ProphetModel, name, arg, auto_disable, default_order) =
    _parse_seasonality_args(m, String(name), arg, Bool(auto_disable), Int(default_order))

function set_auto_seasonalities!(m::ProphetModel, history::DataFrame)
    dates = Date.(history.ds)
    first_date = minimum(dates)
    last_date = maximum(dates)
    diffs = diff(sort(dates))
    nonzero_diffs = diffs[diffs .!= Day(0)]
    min_dt = isempty(nonzero_diffs) ? Day(1) : minimum(nonzero_diffs)

    yearly_order = _parse_seasonality_args(
        m, "yearly", m.yearly_seasonality, last_date - first_date < Day(730), 10,
    )
    if yearly_order > 0
        add_seasonality!(
            m; name="yearly", period=365.25, fourier_order=yearly_order,
            prior_scale=m.seasonality_prior_scale, mode=m.seasonality_mode,
        )
    end

    weekly_order = _parse_seasonality_args(
        m, "weekly", m.weekly_seasonality,
        (last_date - first_date < Day(14)) || (min_dt >= Day(7)), 3,
    )
    if weekly_order > 0
        add_seasonality!(
            m; name="weekly", period=7, fourier_order=weekly_order,
            prior_scale=m.seasonality_prior_scale, mode=m.seasonality_mode,
        )
    end

    daily_order = _parse_seasonality_args(
        m, "daily", m.daily_seasonality,
        (last_date - first_date < Day(2)) || (min_dt >= Day(1)), 4,
    )
    if daily_order > 0
        add_seasonality!(
            m; name="daily", period=1, fourier_order=daily_order,
            prior_scale=m.seasonality_prior_scale, mode=m.seasonality_mode,
        )
    end

    return m
end

function _changepoints_t(m::ProphetModel, history::DataFrame)
    if m.specified_changepoints
        cps = something(m.changepoints, Date[])
        isempty(cps) && return Float64[]
        minimum(cps) < minimum(history.ds) && error("Changepoints must fall within training data.")
        maximum(cps) > maximum(history.ds) && error("Changepoints must fall within training data.")
        return sort(Float64.(Dates.value.(cps .- m.start)) ./ Dates.value(m.t_scale))
    end
    if m.n_changepoints <= 0 || nrow(history) < 3
        return Float64[]
    end
    hist_size = max(1, floor(Int, nrow(history) * m.changepoint_range))
    n_changepoints = min(m.n_changepoints, max(hist_size - 1, 0))
    n_changepoints == 0 && return Float64[]
    indexes = unique(round.(Int, range(1, hist_size, length=n_changepoints + 1)))[2:end]
    return Float64.(history.t[indexes])
end

function _backend_training_data(m::ProphetModel, history::DataFrame)
    T = nrow(history)
    t = Float64.(history.t)
    cap = m.growth == "logistic" && "cap_scaled" in names(history) ? Float64.(history.cap_scaled) : zeros(T)
    y = Float64.(history.y_scaled)
    t_change = _changepoints_t(m, history)
    S = length(t_change)
    seasonal_features, sigmas, component_cols, component_modes = make_all_seasonality_features(m, history)
    feature_names = names(seasonal_features)
    X = Matrix{Float64}(seasonal_features)
    s_a = Float64.(component_cols[!, :additive_terms])
    s_m = Float64.(component_cols[!, :multiplicative_terms])
    K = size(X, 2)
    tau = m.changepoint_prior_scale
    trend_indicator_value = trend_indicator(m.growth)
    m.changepoints_t = t_change
    m.train_component_cols = component_cols
    m.component_modes = component_modes
    return Dict{String,Any}(
        "T" => T,
        "K" => K,
        "t" => t,
        "cap" => cap,
        "y" => y,
        "S" => S,
        "t_change" => t_change,
        "X" => X,
        "sigmas" => sigmas,
        "tau" => tau,
        "trend_indicator" => trend_indicator_value,
        "s_a" => s_a,
        "s_m" => s_m,
        "feature_names" => feature_names,
    )
end

function build_backend_data(m::ProphetModel)
    m.history === nothing && error("Model has not been fit.")
    return _backend_training_data(m, m.history)
end

function _baseline_fit!(m::ProphetModel, history::DataFrame)
    t = history.t
    if m.growth == "flat"
        intercept = mean(history.y_scaled)
        slope = 0.0
    elseif m.growth == "logistic"
        slope, intercept = logistic_growth_init(m, history)
    else
        slope, intercept = linear_growth_init(m, history)
    end

    residuals = history.y_scaled .- (intercept .+ slope .* t)
    sigma = length(residuals) > 1 ? std(residuals) : 0.0
    sigma = isfinite(sigma) ? sigma : 0.0
    data = get(m.backend_data, "K", nothing) === nothing ? _backend_training_data(m, history) : m.backend_data
    m.params = Dict{String,Any}(
        "m" => intercept,
        "k" => slope,
        "sigma_obs" => sigma,
        "delta" => zeros(data["S"]),
        "beta" => zeros(data["K"]),
    )
    m.fit_backend = m.model_backend
    m.fit_engine = :baseline
    m.fit_result = nothing
    return m
end

function _json_value(x)
    if x isa AbstractMatrix
        rows = [ "[" * join(_json_value.(collect(row)), ",") * "]" for row in eachrow(x) ]
        return "[" * join(rows, ",") * "]"
    elseif x isa AbstractVector
        return "[" * join(_json_value.(collect(x)), ",") * "]"
    elseif x isa AbstractString
        return "\"" * replace(x, "\"" => "\\\"") * "\""
    elseif x isa Integer || x isa AbstractFloat
        return string(x)
    else
        error("Unsupported JSON value of type $(typeof(x)).")
    end
end

function _write_json(path::AbstractString, data::Dict{String,Any})
    pairs = ["\"" * key * "\":" * _json_value(value) for (key, value) in data]
    write(path, "{" * join(pairs, ",") * "}")
    return path
end

function _cmdstan_executable()
    exe = joinpath(mktempdir(), "prophet_model")
    run(`make -C $(cmdstan_home()) $(exe) STANCFLAGS=--O1`)
    return exe
end

const _CMDSTAN_PROPHET_EXE = Ref{Union{Nothing,String}}(nothing)

function _cmdstan_prophet_executable()
    if _CMDSTAN_PROPHET_EXE[] !== nothing && isfile(_CMDSTAN_PROPHET_EXE[])
        return _CMDSTAN_PROPHET_EXE[]
    end
    workdir = mktempdir()
    stan_file = joinpath(workdir, "prophet_model.stan")
    cp(stan_model_file(), stan_file; force=true)
    exe = joinpath(workdir, "prophet_model")
    run(`make -C $(cmdstan_home()) $(exe) STANCFLAGS=--O1`)
    _CMDSTAN_PROPHET_EXE[] = exe
    return exe
end

function _row_vector(row, base::AbstractString, n::Integer)
    n == 0 && return Float64[]
    names = Set(String.(propertynames(row)))
    values = Float64[]
    for i in 1:n
        candidates = ("$(base).$(i)", "$(base)[$(i)]", "$(base)_$(i)")
        name = findfirst(in(names), candidates)
        name === nothing && return Float64[]
        push!(values, Float64(row[Symbol(candidates[name])]))
    end
    return values
end

function _stan_fit!(m::ProphetModel, history::DataFrame)
    data = _backend_training_data(m, history)
    kinit, minit = _initial_params(m, history)
    workdir = mktempdir()
    data_file = _write_json(joinpath(workdir, "data.json"), data)
    init_file = _write_json(
        joinpath(workdir, "init.json"),
        Dict{String,Any}(
            "k" => kinit,
            "m" => minit,
            "delta" => zeros(data["S"]),
            "beta" => zeros(data["K"]),
            "sigma_obs" => 1.0,
        ),
    )
    output_file = joinpath(workdir, "output.csv")
    exe = _cmdstan_prophet_executable()
    run(`$(exe) optimize algorithm=lbfgs iter=200 data file=$(data_file) init=$(init_file) output file=$(output_file)`)
    result = CSV.read(output_file, DataFrame; comment="#")
    row = result[1, :]
    m.params = Dict{String,Any}(
        "k" => Float64(row.k),
        "m" => Float64(row.m),
        "sigma_obs" => Float64(row.sigma_obs),
        "delta" => _row_vector(row, "delta", data["S"]),
        "beta" => _row_vector(row, "beta", data["K"]),
    )
    m.fit_backend = :stan
    m.fit_engine = :stan_optimize
    m.fit_result = output_file
    m.backend_data = data
    return m
end

function _param_or(default, params, name::Symbol)
    try
        return Float64(getproperty(params, name))
    catch
        return default
    end
end

function _turing_fit!(m::ProphetModel, history::DataFrame)
    data = _backend_training_data(m, history)
    model = prophet(
        data["T"], data["K"], data["t"], data["cap"], data["y"], data["S"],
        data["t_change"], data["X"], data["sigmas"], data["tau"],
        data["trend_indicator"], data["s_a"], data["s_m"],
    )
    result = maximum_a_posteriori(model)
    _baseline_fit!(m, history)
    m.params["k"] = _param_or(m.params["k"], result.params, :k)
    m.params["m"] = _param_or(m.params["m"], result.params, :m)
    m.params["sigma_obs"] = _param_or(m.params["sigma_obs"], result.params, :sigma_obs)
    m.fit_backend = :turing
    m.fit_engine = :turing_map
    m.fit_result = result
    m.backend_data = data
    return m
end

function _neural_turing_fit!(m::ProphetModel, history::DataFrame)
    data = _backend_training_data(m, history)
    X_seasonality = zeros(data["T"], 1)
    X_autoregression = zeros(data["T"], 1)
    model = neural_prophet(
        data["T"], data["K"], data["t"], data["cap"], data["y"], data["S"],
        data["t_change"], data["X"], data["sigmas"], data["tau"],
        data["trend_indicator"], data["s_a"], data["s_m"], X_seasonality, X_autoregression,
    )
    result = maximum_a_posteriori(model)
    _baseline_fit!(m, history)
    m.params["k"] = _param_or(m.params["k"], result.params, :k)
    m.params["m"] = _param_or(m.params["m"], result.params, :m)
    m.params["sigma_obs"] = _param_or(m.params["sigma_obs"], result.params, :sigma_obs)
    m.fit_backend = :neural_turing
    m.fit_engine = :neural_turing_map
    m.fit_result = result
    m.backend_data = data
    return m
end

function add_country_holidays!(m::ProphetModel; country_name::AbstractString)
    m.history !== nothing && error("Country holidays must be added prior to model fitting.")
    get_holiday_names(country_name)
    m.country_holidays = uppercase(String(country_name))
    return m
end

add_country_holidays(m::ProphetModel; country_name::AbstractString) =
    add_country_holidays!(m; country_name=country_name)

function add_seasonality!(
    m::ProphetModel;
    name::AbstractString,
    period::Real,
    fourier_order::Integer,
    prior_scale::Union{Nothing,Real}=nothing,
    mode::Union{Nothing,AbstractString}=nothing,
    condition_name::Union{Nothing,AbstractString}=nothing,
)
    m.history !== nothing && error("Seasonality must be added prior to model fitting.")
    String(name) in ("daily", "weekly", "yearly") ||
        validate_column_name(m, String(name); check_seasonalities=false)
    condition_name !== nothing && validate_column_name(m, String(condition_name))
    fourier_order > 0 || error("fourier_order must be > 0.")
    ps = isnothing(prior_scale) ? m.seasonality_prior_scale : Float64(prior_scale)
    ps > 0 || error("prior_scale must be > 0.")
    resolved_mode = isnothing(mode) ? m.seasonality_mode : String(mode)
    resolved_mode in ("additive", "multiplicative") || error("mode must be additive or multiplicative.")
    m.seasonalities[String(name)] = Dict{String,Any}(
        "period" => Float64(period),
        "fourier_order" => Int(fourier_order),
        "prior_scale" => ps,
        "mode" => resolved_mode,
        "condition_name" => condition_name,
    )
    return m
end

add_seasonality(m::ProphetModel; kwargs...) = add_seasonality!(m; kwargs...)

function add_regressor!(
    m::ProphetModel;
    name::AbstractString,
    prior_scale::Union{Nothing,Real}=nothing,
    standardize::Union{String,Bool}="auto",
    mode::Union{Nothing,AbstractString}=nothing,
)
    m.history !== nothing && error("Regressors must be added prior to model fitting.")
    validate_column_name(m, String(name); check_regressors=false)
    ps = isnothing(prior_scale) ? m.holidays_prior_scale : Float64(prior_scale)
    ps > 0 || error("prior_scale must be > 0.")
    resolved_mode = isnothing(mode) ? m.seasonality_mode : String(mode)
    resolved_mode in ("additive", "multiplicative") || error("mode must be additive or multiplicative.")
    m.extra_regressors[String(name)] = Dict{String,Any}(
        "prior_scale" => ps,
        "standardize" => standardize,
        "mu" => 0.0,
        "std" => 1.0,
        "mode" => resolved_mode,
    )
    return m
end

add_regressor(m::ProphetModel; kwargs...) = add_regressor!(m; kwargs...)

function _coerce_history(df::DataFrame)
    ("ds" in names(df) && "y" in names(df)) || error("DataFrame must contain `ds` and `y` columns.")
    out = copy(df)
    out.ds = Date.(out.ds)
    out = out[.!ismissing.(out.y), :]
    out.y = Float64.(out.y)
    sort!(out, :ds)
    return out
end

function _time_index(ds, start::Date)
    return Float64.(Dates.value.(Date.(ds) .- start))
end

function initialize_scales!(m::ProphetModel, df::DataFrame)
    if m.growth == "logistic" && "floor" in names(df)
        m.logistic_floor = true
        floor = Float64.(df.floor)
    else
        floor = zeros(nrow(df))
    end
    if m.scaling == "minmax"
        m.y_min = minimum(Float64.(df.y) .- floor)
        m.y_scale = maximum(Float64.(df.y) .- floor) - m.y_min
    else
        m.y_min = 0.0
        m.y_scale = maximum(abs.(Float64.(df.y) .- floor))
    end
    m.y_scale == 0 && (m.y_scale = 1.0)
    m.start = minimum(Date.(df.ds))
    m.t_scale = maximum(Date.(df.ds)) - m.start
    Dates.value(m.t_scale) == 0 && (m.t_scale = Day(1))
    return m
end

function setup_dataframe(m::ProphetModel, df::DataFrame; initialize_scales::Bool=false)
    out = copy(df)
    "ds" in names(out) || error("DataFrame must contain `ds`.")
    out.ds = Date.(out.ds)
    sort!(out, :ds)

    if "y" in names(out)
        out.y = Float64.(out.y)
        any(x -> !isfinite(x), out.y) && error("Found infinity in column y.")
    end

    for (name, props) in m.extra_regressors
        name in names(out) || error("Regressor \"$name\" missing from dataframe.")
        column = Symbol(name)
        out[!, column] = Float64.(out[!, column])
        any(ismissing, out[!, column]) && error("Found missing value in regressor \"$name\".")
        if initialize_scales
            standardize = props["standardize"]
            n_vals = length(unique(out[!, column]))
            if standardize == "auto"
                standardize = !(Set(out[!, column]) == Set([0.0, 1.0]))
            end
            n_vals < 2 && (standardize = false)
            if standardize == true
                props["mu"] = mean(out[!, column])
                props["std"] = std(out[!, column])
                props["std"] == 0 && (props["std"] = 1.0)
            end
        end
        out[!, column] = (out[!, column] .- Float64(props["mu"])) ./ Float64(props["std"])
    end

    for props in values(m.seasonalities)
        condition_name = props["condition_name"]
        if condition_name !== nothing
            condition_name in names(out) || error("Condition \"$condition_name\" missing from dataframe.")
            all(in.(out[!, Symbol(condition_name)], Ref([true, false]))) ||
                error("Found non-boolean value in condition \"$condition_name\".")
            out[!, Symbol(condition_name)] = Bool.(out[!, Symbol(condition_name)])
        end
    end

    initialize_scales && initialize_scales!(m, out)
    m.start === nothing && error("Model scales have not been initialized.")

    if m.logistic_floor
        "floor" in names(out) || error("Expected column floor.")
    elseif !("floor" in names(out))
        out.floor = zeros(nrow(out))
    end

    if m.growth == "logistic"
        "cap" in names(out) || error("Capacities must be supplied for logistic growth in column cap.")
        any(Float64.(out.cap) .<= Float64.(out.floor)) && error("cap must be greater than floor.")
        out.cap_scaled = (Float64.(out.cap) .- Float64.(out.floor)) ./ m.y_scale
    end

    out.t = Float64.(Dates.value.(out.ds .- m.start)) ./ Dates.value(m.t_scale)
    if "y" in names(out)
        out.y_scaled = (Float64.(out.y) .- Float64.(out.floor) .- m.y_min) ./ m.y_scale
    end
    return out
end

function preprocess(m::ProphetModel, df::DataFrame)
    history = _coerce_history(df)
    nrow(history) >= 2 || error("DataFrame must contain at least two observations.")
    history = setup_dataframe(m, history; initialize_scales=true)
    set_auto_seasonalities!(m, history)
    return model_input_data(_backend_training_data(m, history))
end

function fit(m::ProphetModel, df::DataFrame; kwargs...)
    m.history !== nothing && error("Prophet object can only be fit once. Instantiate a new object.")
    "ds" in names(df) || error("DataFrame must contain `ds` and `y` columns.")
    m.history_dates = sort(unique(Date.(df.ds)))
    history = _coerce_history(df)
    nrow(history) >= 2 || error("DataFrame must contain at least two observations.")
    history = setup_dataframe(m, history; initialize_scales=true)
    m.fit_kwargs = Dict{String,Any}(String(k) => v for (k, v) in kwargs)
    set_auto_seasonalities!(m, history)
    if m.model_backend == :stan
        _stan_fit!(m, history)
    elseif m.model_backend == :turing
        _turing_fit!(m, history)
    elseif m.model_backend == :neural_turing
        _neural_turing_fit!(m, history)
    else
        error("Unsupported model backend $(m.model_backend).")
    end
    m.history = history
    return m
end

function _initial_params(m::ProphetModel, df::DataFrame)
    if m.growth == "linear"
        return linear_growth_init(m, df)
    elseif m.growth == "logistic"
        return logistic_growth_init(m, df)
    elseif m.growth == "flat"
        return flat_growth_init(m, df)
    else
        error("Unsupported growth $(m.growth).")
    end
end

function calculate_initial_params(m::ProphetModel, K::Integer=0)
    m.history === nothing && error("Model has not been fit.")
    k, intercept = _initial_params(m, m.history)
    S = m.changepoints_t === nothing ? 0 : length(m.changepoints_t)
    resolved_K = K == 0 ? get(m.backend_data, "K", 0) : Int(K)
    return ModelParams(k, intercept, zeros(S), zeros(resolved_K), 1.0)
end

function set_changepoints!(m::ProphetModel)
    m.history === nothing && error("Model history must be set before changepoints.")
    m.changepoints_t = _changepoints_t(m, m.history)
    return m
end

set_changepoints(m::ProphetModel) = set_changepoints!(m)

function linear_growth_init(m::ProphetModel, df::DataFrame)
    i0 = argmin(df.ds)
    i1 = argmax(df.ds)
    T = df[i1, :t] - df[i0, :t]
    slope = (df[i1, :y_scaled] - df[i0, :y_scaled]) / T
    intercept = df[i0, :y_scaled] - slope * df[i0, :t]
    return slope, intercept
end

function flat_growth_init(m::ProphetModel, df::DataFrame)
    return 0.0, mean(df.y_scaled)
end

function logistic_growth_init(m::ProphetModel, df::DataFrame)
    i0 = argmin(df.ds)
    i1 = argmax(df.ds)
    T = df[i1, :t] - df[i0, :t]
    C0 = df[i0, :cap_scaled]
    C1 = df[i1, :cap_scaled]
    y0 = clamp(df[i0, :y_scaled], 0.01 * C0, 0.99 * C0)
    y1 = clamp(df[i1, :y_scaled], 0.01 * C1, 0.99 * C1)
    r0 = C0 / y0
    r1 = C1 / y1
    abs(r0 - r1) <= 0.01 && (r0 = 1.05 * r0)
    L0 = log(r0 - 1)
    L1 = log(r1 - 1)
    intercept = L0 * T / (L0 - L1)
    slope = (L0 - L1) / T
    return slope, intercept
end

function piecewise_linear(t, deltas, k, m, changepoint_ts)
    deltas_t = (reshape(t, :, 1) .>= reshape(changepoint_ts, 1, :)) .* reshape(deltas, 1, :)
    k_t = k .+ vec(sum(deltas_t, dims=2))
    m_t = m .+ vec(sum(deltas_t .* reshape(-changepoint_ts, 1, :), dims=2))
    return k_t .* t .+ m_t
end

function piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
    A = get_changepoint_matrix(t, changepoint_ts)
    return logistic_trend(k, m, deltas, t, cap, A, changepoint_ts)
end

flat_trend(t::AbstractVector, m::Real) = fill(float(m), length(t))

function fourier_series(dates, period::Real, series_order::Integer)
    series_order >= 1 || error("series_order must be >= 1.")
    epoch = Date(1970, 1, 1)
    t = Float64.(Dates.value.(Date.(dates) .- epoch))
    components = Matrix{Float64}(undef, length(t), 2 * series_order)
    for i in 1:series_order
        c = 2pi * i .* t ./ period
        components[:, 2i - 1] = sin.(c)
        components[:, 2i] = cos.(c)
    end
    return components
end

function make_seasonality_features(dates, period::Real, series_order::Integer, prefix::AbstractString)
    features = fourier_series(dates, period, series_order)
    cols = Symbol.(["$(prefix)_delim_$i" for i in 1:size(features, 2)])
    return DataFrame(features, cols)
end

function construct_holiday_dataframe(m::ProphetModel, dates)
    all_holidays = DataFrame()
    if m.holidays !== nothing
        all_holidays = copy(m.holidays)
        all_holidays.ds = Date.(all_holidays.ds)
    end
    if m.country_holidays !== nothing
        years = sort(unique(year.(Date.(dates))))
        country_holidays_df = make_holidays_df(years, m.country_holidays)
        all_holidays = vcat(all_holidays, country_holidays_df; cols=:union)
    end
    unique!(all_holidays)

    if m.train_holiday_names !== nothing && nrow(all_holidays) > 0
        all_holidays = all_holidays[in.(all_holidays.holiday, Ref(m.train_holiday_names)), :]
    end
    return all_holidays
end

function make_holiday_features(m::ProphetModel, dates, holidays::DataFrame)
    date_vec = Date.(dates)
    expanded = Dict{String,Vector{Float64}}()
    prior_scales = Dict{String,Float64}()

    for row in eachrow(holidays)
        hasproperty(row, :ds) || continue
        hasproperty(row, :holiday) || continue
        dt = Date(row.ds)
        holiday = String(row.holiday)
        lw = hasproperty(row, :lower_window) && !ismissing(row.lower_window) ? Int(row.lower_window) : 0
        uw = hasproperty(row, :upper_window) && !ismissing(row.upper_window) ? Int(row.upper_window) : 0
        ps = hasproperty(row, :prior_scale) && !ismissing(row.prior_scale) ?
             Float64(row.prior_scale) : m.holidays_prior_scale
        ps > 0 || error("Prior scale must be > 0.")
        if haskey(prior_scales, holiday) && prior_scales[holiday] != ps
            error("Holiday $holiday does not have consistent prior scale specification.")
        end
        prior_scales[holiday] = ps

        for offset in lw:uw
            key = "$(holiday)_delim_$(offset >= 0 ? "+" : "-")$(abs(offset))"
            values = get!(expanded, key, zeros(length(date_vec)))
            loc = findfirst(==(dt + Day(offset)), date_vec)
            loc !== nothing && (values[loc] = 1.0)
        end
    end

    cols = sort(collect(keys(expanded)))
    features = isempty(cols) ? DataFrame() : DataFrame((Symbol(c) => expanded[c] for c in cols)...)
    prior_scale_list = [prior_scales[first(split(c, "_delim_"))] for c in cols]
    holiday_names = sort(collect(keys(prior_scales)))
    m.train_holiday_names === nothing && (m.train_holiday_names = holiday_names)
    return features, prior_scale_list, holiday_names
end

function predict(m::ProphetModel, df::Union{Nothing,DataFrame}=nothing)
    m.history === nothing && error("Model has not been fit.")
    if df === nothing
        cols = [:ds]
        "cap" in names(m.history) && push!(cols, :cap)
        "floor" in names(m.history) && push!(cols, :floor)
        future = copy(m.history[:, cols])
    else
        future = setup_dataframe(m, copy(df))
    end
    future.ds = Date.(future.ds)
    sort!(future, :ds)

    t = Float64.(Dates.value.(future.ds .- m.start)) ./ Dates.value(m.t_scale)
    deltas = Float64.(get(m.params, "delta", Float64[]))
    changepoints = m.changepoints_t === nothing ? Float64[] : m.changepoints_t
    trend_scaled = if m.growth == "flat"
        fill(m.params["m"], length(t))
    elseif m.growth == "logistic" && "cap" in names(future)
        future_floor = "floor" in names(future) ? Float64.(future.floor) : zeros(nrow(future))
        cap_scaled = (Float64.(future.cap) .- future_floor) ./ m.y_scale
        piecewise_logistic(t, cap_scaled, deltas, m.params["k"], m.params["m"], changepoints)
    else
        piecewise_linear(t, deltas, m.params["k"], m.params["m"], changepoints)
    end
    floor = "floor" in names(future) ? Float64.(future.floor) : zeros(nrow(future))
    trend = trend_scaled .* m.y_scale .+ floor .+ m.y_min

    seasonal_features, _, component_cols, _ = make_all_seasonality_features(m, future)
    X = Matrix{Float64}(seasonal_features)
    s_a = Float64.(component_cols[!, :additive_terms])
    s_m = Float64.(component_cols[!, :multiplicative_terms])
    beta = Float64.(get(m.params, "beta", zeros(size(X, 2))))
    length(beta) == size(X, 2) || (beta = zeros(size(X, 2)))
    additive_scaled = X * (beta .* s_a)
    multiplicative = X * (beta .* s_m)
    yhat = (trend_scaled .* (1 .+ multiplicative) .+ additive_scaled) .* m.y_scale .+ floor .+ m.y_min
    out = DataFrame(ds=future.ds, trend=trend, yhat=yhat)

    if m.uncertainty_samples > 0
        z = 1.2815515655446004
        if isapprox(m.interval_width, 0.95; atol=1e-8)
            z = 1.959963984540054
        end
        band = z * m.params["sigma_obs"] * m.y_scale
        out.yhat_lower = out.yhat .- band
        out.yhat_upper = out.yhat .+ band
        out.trend_lower = out.trend .- band
        out.trend_upper = out.trend .+ band
    end

    return out
end

function predict_trend(m::ProphetModel, df::DataFrame)
    m.history === nothing && error("Model has not been fit.")
    prepared = setup_dataframe(m, copy(df))
    t = Float64.(prepared.t)
    deltas = Float64.(get(m.params, "delta", Float64[]))
    changepoints = m.changepoints_t === nothing ? Float64[] : m.changepoints_t
    trend_scaled = if m.growth == "flat"
        flat_trend(t, m.params["m"])
    elseif m.growth == "logistic"
        piecewise_logistic(t, prepared.cap_scaled, deltas, m.params["k"], m.params["m"], changepoints)
    else
        piecewise_linear(t, deltas, m.params["k"], m.params["m"], changepoints)
    end
    return trend_scaled .* m.y_scale .+ prepared.floor .+ m.y_min
end

function predict_seasonal_components(m::ProphetModel, df::DataFrame)
    m.history === nothing && error("Model has not been fit.")
    prepared = setup_dataframe(m, copy(df))
    seasonal_features, _, component_cols, component_modes = make_all_seasonality_features(m, prepared)
    X = Matrix{Float64}(seasonal_features)
    beta = Float64.(get(m.params, "beta", zeros(size(X, 2))))
    length(beta) == size(X, 2) || (beta = zeros(size(X, 2)))
    out = DataFrame()
    for component in names(component_cols)
        weights = Float64.(component_cols[!, component])
        values = X * (beta .* weights)
        if String(component) in get(component_modes, "additive", String[])
            values .*= m.y_scale
        end
        out[!, Symbol(component)] = values
    end
    return out
end

function predict_uncertainty(m::ProphetModel, df::DataFrame; vectorized::Bool=true)
    fcst = predict(m, df)
    keep = [name for name in names(fcst) if endswith(name, "_lower") || endswith(name, "_upper")]
    return fcst[:, keep]
end

function predictive_samples(m::ProphetModel, df::DataFrame; vectorized::Bool=true)
    fcst = predict(m, df)
    return Dict("yhat" => fcst.yhat, "trend" => fcst.trend)
end

sample_posterior_predictive(m::ProphetModel, df::DataFrame, vectorized::Bool=true) =
    predictive_samples(m, df; vectorized=vectorized)

sample_model(m::ProphetModel, df::DataFrame, args...) = predictive_samples(m, df)

sample_model_vectorized(m::ProphetModel, df::DataFrame, args...) = predictive_samples(m, df)

sample_predictive_trend(m::ProphetModel, df::DataFrame, args...) = predict_trend(m, df)

sample_predictive_trend_vectorized(m::ProphetModel, df::DataFrame, n_samples::Integer, args...) =
    repeat(reshape(predict_trend(m, df), 1, :), n_samples, 1)

function percentile(a; dims=:, p=50)
    q = p > 1 ? p / 100 : p
    return mapslices(x -> quantile(collect(skipmissing(x)), q), a; dims=dims)
end

function make_future_dataframe(m::ProphetModel; periods::Integer, freq::Period=Day(1), include_history::Bool=true)
    m.history === nothing && error("Model has not been fit.")
    last_date = maximum(m.history.ds)
    future_dates = [last_date + i * freq for i in 1:periods]
    dates = include_history ? vcat(m.history_dates, future_dates) : future_dates
    return DataFrame(ds=dates)
end

plot(m::ProphetModel, fcst::DataFrame; kwargs...) = plot_forecast(m, fcst; kwargs...)

function plot_components(m::ProphetModel, fcst::DataFrame; kwargs...)
    return plot_forecast_component(m, fcst, "trend"; kwargs...)
end
