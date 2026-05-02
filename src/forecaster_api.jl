using DataFrames
using Dates
using LinearAlgebra
using Statistics

mutable struct ProphetModel
    growth::String
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
    seasonalities::Dict{String,Dict{String,Any}}
    extra_regressors::Dict{String,Dict{String,Any}}
    train_holiday_names::Union{Nothing,Vector{String}}
end

function ProphetModel(;
    growth::AbstractString="linear",
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
    0 <= changepoint_range <= 1 || error("changepoint_range must be in [0, 1].")
    seasonality_mode in ("additive", "multiplicative") ||
        error("seasonality_mode must be \"additive\" or \"multiplicative\".")
    resolved_holidays_mode = isnothing(holidays_mode) ? String(seasonality_mode) : String(holidays_mode)
    resolved_holidays_mode in ("additive", "multiplicative") ||
        error("holidays_mode must be \"additive\" or \"multiplicative\".")
    return ProphetModel(
        String(growth),
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
        Dict{String,Dict{String,Any}}(),
        Dict{String,Dict{String,Any}}(),
        nothing,
    )
end

const Prophet = ProphetModel

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
