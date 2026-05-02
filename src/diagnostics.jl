import Dagger
using DataFrames
using Dates
using Statistics

function _period(x::Period)
    return x
end

function _period(x::Integer)
    return Day(x)
end

function _period(x::AbstractString)
    parts = split(strip(x))
    length(parts) == 2 || error("Periods must look like \"365 days\".")
    value = parse(Int, parts[1])
    unit = lowercase(parts[2])
    if startswith(unit, "day")
        return Day(value)
    elseif startswith(unit, "week")
        return Week(value)
    elseif startswith(unit, "hour")
        return Hour(value)
    elseif startswith(unit, "minute")
        return Minute(value)
    else
        error("Unsupported period unit \"$unit\".")
    end
end

"""
    generate_cutoffs(df, horizon, initial, period)

Generate historical cutoff dates using Prophet's rolling-origin cross-validation
scheme.
"""
function generate_cutoffs(df::DataFrame, horizon, initial, period)
    h = _period(horizon)
    i = _period(initial)
    p = _period(period)

    cutoff = maximum(Date.(df.ds)) - h
    cutoff < minimum(Date.(df.ds)) && error("Less data than horizon.")

    cutoffs = Date[]
    while cutoff >= minimum(Date.(df.ds)) + i
        push!(cutoffs, cutoff)
        cutoff -= p
        if !any((Date.(df.ds) .> cutoff) .& (Date.(df.ds) .<= cutoff + h)) &&
                cutoff > minimum(Date.(df.ds))
            cutoff = maximum(Date.(df[df.ds .<= cutoff, :ds])) - h
        end
    end

    isempty(cutoffs) && error("Less data than horizon after initial window.")
    return reverse(cutoffs)
end

function _copy_for_cross_validation(m::ProphetModel)
    m2 = ProphetModel(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        yearly_seasonality=m.yearly_seasonality,
        weekly_seasonality=m.weekly_seasonality,
        daily_seasonality=m.daily_seasonality,
        holidays=m.holidays,
        country_holidays=m.country_holidays,
        seasonality_mode=m.seasonality_mode,
        holidays_mode=m.holidays_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
    )
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.extra_regressors = deepcopy(m.extra_regressors)
    return m2
end

function single_cutoff_forecast(df::DataFrame, model::ProphetModel, cutoff::Date, horizon)
    h = _period(horizon)
    history_c = df[Date.(df.ds) .<= cutoff, :]
    nrow(history_c) >= 2 || error("Less than two datapoints before cutoff.")

    m = _copy_for_cross_validation(model)
    fit(m, history_c)

    index_predicted = (Date.(df.ds) .> cutoff) .& (Date.(df.ds) .<= cutoff + h)
    future = df[index_predicted, [:ds]]
    fcst = predict(m, future)
    fcst.y = Float64.(df[index_predicted, :y])
    fcst.cutoff = fill(cutoff, nrow(fcst))
    return fcst
end

"""
    cross_validation(model; horizon, period=nothing, initial=nothing, cutoffs=nothing, parallel=nothing)

Compute simulated historical forecasts. `parallel=:dagger` is the Julia analogue
to Python Prophet's `parallel="dask"` mode: each cutoff forecast is submitted to
Dagger's task scheduler.
"""
function cross_validation(
    model::ProphetModel;
    horizon,
    period=nothing,
    initial=nothing,
    cutoffs=nothing,
    parallel=nothing,
    disable_tqdm=false,
)
    model.history === nothing && error("Model has not been fit.")
    df = copy(model.history)
    h = _period(horizon)
    p = isnothing(period) ? Day(max(1, Int(floor(Dates.value(h) / 2)))) : _period(period)
    i = isnothing(initial) ? 3 * h : _period(initial)
    resolved_cutoffs = isnothing(cutoffs) ? generate_cutoffs(df, h, i, p) : Date.(cutoffs)

    if parallel in (:dagger, "dagger", "dask")
        tasks = [Dagger.@spawn single_cutoff_forecast(df, model, cutoff, h) for cutoff in resolved_cutoffs]
        return vcat(fetch.(tasks)...)
    elseif parallel in (:threads, "threads")
        tasks = [Threads.@spawn single_cutoff_forecast(df, model, cutoff, h) for cutoff in resolved_cutoffs]
        return vcat(fetch.(tasks)...)
    elseif isnothing(parallel) || parallel == false
        return vcat([single_cutoff_forecast(df, model, cutoff, h) for cutoff in resolved_cutoffs]...)
    else
        error("Unsupported parallel mode $parallel. Use nothing, :threads, or :dagger.")
    end
end

function _horizon(df::DataFrame)
    return Date.(df.ds) .- Date.(df.cutoff)
end

function _metric_frame(df::DataFrame, metric::Symbol, values)
    return DataFrame(horizon=_horizon(df), metric => Float64.(values))
end

mse(df::DataFrame) = _metric_frame(df, :mse, (df.y .- df.yhat) .^ 2)
rmse(df::DataFrame) = _metric_frame(df, :rmse, sqrt.((df.y .- df.yhat) .^ 2))
mae(df::DataFrame) = _metric_frame(df, :mae, abs.(df.y .- df.yhat))
mape(df::DataFrame) = _metric_frame(df, :mape, abs.((df.y .- df.yhat) ./ df.y))
mdape(df::DataFrame) = DataFrame(horizon=[maximum(_horizon(df))], mdape=[median(abs.((df.y .- df.yhat) ./ df.y))])
smape(df::DataFrame) = _metric_frame(df, :smape, abs.(df.y .- df.yhat) ./ ((abs.(df.y) .+ abs.(df.yhat)) ./ 2))

function coverage(df::DataFrame)
    ("yhat_lower" in names(df) && "yhat_upper" in names(df)) ||
        error("coverage requires yhat_lower and yhat_upper columns.")
    return _metric_frame(df, :coverage, (df.yhat_lower .<= df.y) .& (df.y .<= df.yhat_upper))
end

function _aggregate_metric(metric_df::DataFrame, metric::Symbol, rolling_window)
    sort!(metric_df, :horizon)
    if rolling_window < 0
        return metric_df
    end
    grouped = combine(groupby(metric_df, :horizon), metric => mean => metric)
    return grouped
end

"""
    performance_metrics(df; metrics=nothing, rolling_window=0.1)

Return Prophet-style forecast diagnostics for a cross-validation DataFrame.
Supported metrics are `mse`, `rmse`, `mae`, `mape`, `mdape`, `smape`, and
`coverage`.
"""
function performance_metrics(df::DataFrame; metrics=nothing, rolling_window=0.1, monthly=false)
    valid = [:mse, :rmse, :mae, :mape, :mdape, :smape, :coverage]
    selected = isnothing(metrics) ? valid : Symbol.(metrics)
    if !("yhat_lower" in names(df) && "yhat_upper" in names(df))
        selected = filter(!=(:coverage), selected)
    end
    all(in(valid), selected) || error("Valid metrics are $(String.(valid)).")
    isempty(selected) && return DataFrame()

    results = Dict(
        :mse => mse,
        :rmse => rmse,
        :mae => mae,
        :mape => mape,
        :mdape => mdape,
        :smape => smape,
        :coverage => coverage,
    )

    frames = [_aggregate_metric(results[metric](df), metric, rolling_window) for metric in selected]
    out = frames[1]
    for frame in frames[2:end]
        out = outerjoin(out, frame, on=:horizon)
    end
    sort!(out, :horizon)
    return out
end
