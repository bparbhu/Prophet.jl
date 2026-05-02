using CSV
using DataFrames
using Dates
using LinearAlgebra
using Statistics
using Turing

mutable struct ProphetModel
    growth::String
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
    logistic_floor::Bool
    history::Union{Nothing,DataFrame}
    history_dates::Union{Nothing,Vector{Date}}
    start::Union{Nothing,Date}
    y_min::Union{Nothing,Float64}
    y_scale::Union{Nothing,Float64}
    t_scale::Union{Nothing,Period}
    params::Dict{String,Float64}
    fit_backend::Union{Nothing,Symbol}
    fit_engine::Union{Nothing,Symbol}
    fit_result::Any
    backend_data::Dict{String,Any}
    seasonalities::Dict{String,Dict{String,Any}}
    extra_regressors::Dict{String,Dict{String,Any}}
    train_holiday_names::Union{Nothing,Vector{String}}
    changepoints_t::Union{Nothing,Vector{Float64}}
end

function ProphetModel(;
    growth::AbstractString="linear",
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
    return ProphetModel(
        String(growth),
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
        false,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        Dict{String,Float64}(),
        nothing,
        nothing,
        nothing,
        Dict{String,Any}(),
        Dict{String,Dict{String,Any}}(),
        Dict{String,Dict{String,Any}}(),
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

function set_model_backend!(m::ProphetModel, backend)
    m.history !== nothing && error("Model backend must be selected before fitting.")
    m.model_backend = _normalize_model_backend(backend, nothing)
    return m
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

    if isempty(feature_frames)
        return zeros(nrow(history), 1), [1.0], [0.0], [0.0], String[]
    end

    features = hcat(feature_frames...; makeunique=true)
    X = Matrix{Float64}(features)
    s_a = Float64.(modes .== "additive")
    s_m = Float64.(modes .== "multiplicative")
    return X, prior_scales, s_a, s_m, names(features)
end

function _changepoints_t(m::ProphetModel, history::DataFrame)
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
    X, sigmas, s_a, s_m, feature_names = _seasonality_design_matrix(m, history)
    K = size(X, 2)
    tau = m.changepoint_prior_scale
    trend_indicator = Dict("linear" => 0, "logistic" => 1, "flat" => 2)[m.growth]
    m.changepoints_t = t_change
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
        "trend_indicator" => trend_indicator,
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
    else
        X = hcat(ones(length(t)), t)
        intercept, slope = X \ history.y_scaled
    end

    residuals = history.y_scaled .- (intercept .+ slope .* t)
    sigma = length(residuals) > 1 ? std(residuals) : 0.0
    sigma = isfinite(sigma) ? sigma : 0.0
    m.params = Dict("m" => intercept, "k" => slope, "sigma_obs" => sigma)
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

function _stan_fit!(m::ProphetModel, history::DataFrame)
    data = _backend_training_data(m, history)
    workdir = mktempdir()
    data_file = _write_json(joinpath(workdir, "data.json"), data)
    init_file = _write_json(
        joinpath(workdir, "init.json"),
        Dict{String,Any}(
            "k" => 0.0,
            "m" => mean(history.y_scaled),
            "delta" => Float64[],
            "beta" => [0.0],
            "sigma_obs" => 0.1,
        ),
    )
    output_file = joinpath(workdir, "output.csv")
    exe = _cmdstan_prophet_executable()
    run(`$(exe) optimize algorithm=lbfgs iter=200 data file=$(data_file) init=$(init_file) output file=$(output_file)`)
    result = CSV.read(output_file, DataFrame; comment="#")
    row = result[1, :]
    m.params = Dict(
        "k" => Float64(row.k),
        "m" => Float64(row.m),
        "sigma_obs" => Float64(row.sigma_obs),
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
    m.y_min = 0.0
    m.y_scale = maximum(abs.(Float64.(df.y) .- floor))
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
        out.y_scaled = (Float64.(out.y) .- Float64.(out.floor)) ./ m.y_scale
    end
    return out
end

function fit(m::ProphetModel, df::DataFrame)
    history = _coerce_history(df)
    nrow(history) >= 2 || error("DataFrame must contain at least two observations.")
    history = setup_dataframe(m, history; initialize_scales=true)
    m.history_dates = sort(unique(history.ds))
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

function linear_growth_init(m::ProphetModel, df::DataFrame)
    X = hcat(ones(nrow(df)), df.t)
    intercept, slope = X \ df.y_scaled
    return slope, intercept
end

function flat_growth_init(m::ProphetModel, df::DataFrame)
    return 0.0, mean(df.y_scaled)
end

function logistic_growth_init(m::ProphetModel, df::DataFrame)
    return linear_growth_init(m, df)
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
    features = isempty(cols) ? DataFrame() : DataFrame([Symbol(c) => expanded[c] for c in cols])
    prior_scale_list = [prior_scales[first(split(c, "_delim_"))] for c in cols]
    holiday_names = sort(collect(keys(prior_scales)))
    m.train_holiday_names === nothing && (m.train_holiday_names = holiday_names)
    return features, prior_scale_list, holiday_names
end

function predict(m::ProphetModel, df::Union{Nothing,DataFrame}=nothing)
    m.history === nothing && error("Model has not been fit.")
    future = df === nothing ? copy(m.history[:, [:ds]]) : copy(df)
    future.ds = Date.(future.ds)
    sort!(future, :ds)

    t = Float64.(Dates.value.(future.ds .- m.start)) ./ Dates.value(m.t_scale)
    trend_scaled = if m.growth == "flat"
        fill(m.params["m"], length(t))
    elseif m.growth == "logistic" && "cap" in names(future)
        future_floor = "floor" in names(future) ? Float64.(future.floor) : zeros(nrow(future))
        cap_scaled = (Float64.(future.cap) .- future_floor) ./ m.y_scale
        piecewise_logistic(t, cap_scaled, Float64[], m.params["k"], m.params["m"], Float64[])
    else
        m.params["m"] .+ m.params["k"] .* t
    end
    floor = "floor" in names(future) ? Float64.(future.floor) : zeros(nrow(future))
    trend = trend_scaled .* m.y_scale .+ floor
    out = DataFrame(ds=future.ds, trend=trend, yhat=trend)

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

function make_future_dataframe(m::ProphetModel; periods::Integer, freq::Period=Day(1), include_history::Bool=true)
    m.history === nothing && error("Model has not been fit.")
    last_date = maximum(m.history.ds)
    future_dates = [last_date + i * freq for i in 1:periods]
    dates = include_history ? vcat(m.history.ds, future_dates) : future_dates
    return DataFrame(ds=dates)
end

plot(m::ProphetModel, fcst::DataFrame; kwargs...) = plot_forecast(m, fcst; kwargs...)

function plot_components(m::ProphetModel, fcst::DataFrame; kwargs...)
    return plot_forecast_component(m, fcst, "trend"; kwargs...)
end
