using DataFrames, Logging, Dates, Statistics, ProgressMeter, Random, Statistics, LinearAlgebra

logger = Logging.current_logger()

function generate_cutoffs(df, horizon, initial, period)
    cutoff = maximum(df.ds) - horizon
    if cutoff < minimum(df.ds)
        throw(ArgumentError("Less data than horizon."))
    end
    result = [cutoff]
    while last(result) >= minimum(df.ds) + initial
        cutoff -= period
        if !any((df.ds .> cutoff) .& (df.ds .<= cutoff + horizon))
            if cutoff > minimum(df.ds)
                closest_date = maximum(df[df.ds .<= cutoff, :ds])
                cutoff = closest_date - horizon
            end
        end
        push!(result, cutoff)
    end
    result = result[1:end-1]
    if isempty(result)
        throw(ArgumentError(
            "Less data than horizon after initial window. " *
            "Make horizon or initial shorter."
        ))
    end
    Logging.info(logger, "Making {} forecasts with cutoffs between {} and {}",
        length(result), last(result), first(result))
    return reverse(result)
end

function cross_validation(model, horizon, period=nothing, initial=nothing, parallel=nothing, cutoffs=nothing, disable_tqdm=false)
    if isnothing(model.history)
        error("Model has not been fit. Fitting the model provides contextual parameters for cross validation.")
    end

    df = deepcopy(model.history)
    horizon = Dates.Millisecond(horizon)
    predict_columns = [:ds, :yhat]
    if model.uncertainty_samples
        append!(predict_columns, [:yhat_lower, :yhat_upper])
    end

    period_max = 0.0
    for s in values(model.seasonalities)
        period_max = max(period_max, s.period)
    end
    seasonality_dt = Dates.Millisecond(period_max)

    if isnothing(cutoffs)
        period = isnothing(period) ? 0.5 * horizon : Dates.Millisecond(period)
        initial = isnothing(initial) ? max(3 * horizon, seasonality_dt) : Dates.Millisecond(initial)
        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else
        if minimum(cutoffs) <= minimum(df.ds)
            error("Minimum cutoff value is not strictly greater than min date in history")
        end
        end_date_minus_horizon = maximum(df.ds) - horizon
        if maximum(cutoffs) > end_date_minus_horizon
            error("Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining")
        end
        initial = cutoffs[1] - minimum(df.ds)
    end

    if initial < seasonality_dt
        msg = "Seasonality has period of $(period_max) days which is larger than initial window. Consider increasing initial."
        @warn msg
    end

    if !isnothing(parallel)
        error("Parallel processing is not supported in this Julia implementation.")
    else
        if disable_tqdm
            predicts = [single_cutoff_forecast(df, model, cutoff, horizon, predict_columns) for cutoff in cutoffs]
        else
            predicts = []
            p = Progress(length(cutoffs), desc="Computing forecasts: ")
            for cutoff in cutoffs
                push!(predicts, single_cutoff_forecast(df, model, cutoff, horizon, predict_columns))
                next!(p)
            end
        end
    end
    return vcat(predicts...)
end

function single_cutoff_forecast(df, model, cutoff, horizon, predict_columns)
    m = prophet_copy(model, cutoff)
    history_c = df[df.ds .<= cutoff, :]
    if nrow(history_c) < 2
        throw(ArgumentError(
            "Less than two datapoints before cutoff. " *
            "Increase initial window."
        ))
    end
    fit!(m, history_c, model.fit_kwargs...)
    index_predicted = (df.ds .> cutoff) .& (df.ds .<= cutoff + horizon)
    columns = ["ds"]
    if m.growth == "logistic"
        push!(columns, "cap")
        if m.logistic_floor
            push!(columns, "floor")
        end
    end
    append!(columns, keys(m.extra_regressors))
    append!(columns, [
        props["condition_name"]
        for props in values(m.seasonalities)
        if props["condition_name"] !== nothing
    ])
    yhat = predict(m, df[index_predicted, columns])
    return hcat(
        yhat[:, predict_columns],
        df[index_predicted, :y],
        DataFrame(cutoff=fill(cutoff, nrow(yhat)))
    )
end


function prophet_copy(m, cutoff=nothing)
    if m.history === nothing
        throw(ArgumentError("This is for copying a fitted Prophet object."))
    end

    if m.specified_changepoints
        changepoints = m.changepoints
        if cutoff !== nothing
            last_history_date = maximum(m.history[m.history.ds .<= cutoff, :ds])
            changepoints = changepoints[changepoints .< last_history_date]
        end
    else
        changepoints = nothing
    end

    m2 = m.__type__(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        changepoints=changepoints,
        yearly_seasonality=false,
        weekly_seasonality=false,
        daily_seasonality=false,
        holidays=m.holidays,
        seasonality_mode=m.seasonality_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
        stan_backend=(
            m.stan_backend !== nothing ? m.stan_backend.get_type() : nothing
        ),
    )
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.country_holidays = deepcopy(m.country_holidays)
    return m2
end


function performance_metrics(df, metrics=nothing, rolling_window=0.1, monthly=false)
    valid_metrics = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    if metrics === nothing
        metrics = valid_metrics
    end
    if !("yhat_lower" in names(df) || "yhat_upper" in names(df)) && "coverage" in metrics
        deleteat!(metrics, findfirst(==(coverage), metrics))
    end
    if length(unique(metrics)) != length(metrics)
        throw(ArgumentError("Input metrics must be a list of unique values"))
    end
    if !issubset(Set(metrics), Set(valid_metrics))
        throw(ArgumentError("Valid values for metrics are: $(valid_metrics)"))
    end
    df_m = copy(df)
    if monthly
        df_m.horizon = month.(df_m.ds) .- month.(df_m.cutoff)
    else
        df_m.horizon = df_m.ds .- df_m.cutoff
    end
    sort!(df_m, :horizon)
    if "mape" in metrics && minimum(abs.(df_m.y)) < 1e-8
        Logging.logger.info("Skipping MAPE because y close to 0")
        deleteat!(metrics, findfirst(==(mape), metrics))
    end
    if length(metrics) == 0
        return nothing
    end
    w = trunc(Int, rolling_window * nrow(df_m))
    if w >= 0
        w = max(w, 1)
        w = min(w, nrow(df_m))
    end
    # Compute all metrics
    dfs = Dict()
    for metric in metrics
        dfs[metric] = eval(metric)(df_m, w)
    end
    res = dfs[metrics[1]]
    for i in 2:length(metrics)
        res_m = dfs[metrics[i]]
        @assert res.horizon == res_m.horizon
        res[!, metrics[i]] = res_m[!, metrics[i]]
    end
    return res
end


function rolling_mean_by_h(x, h, w, name)
    df = DataFrame(x=x, h=h)
    df2 = combine(groupby(df, :h), :x => sum => :x_sum, :x => length => :x_count)
    sort!(df2, :h)
    xs = df2.x_sum
    ns = df2.x_count
    hs = df2.h

    trailing_i = nrow(df2) - 1
    x_sum = 0
    n_sum = 0
    res_x = Array{Float64}(undef, nrow(df2))

    for i in (nrow(df2) - 1):-1:0
        x_sum += xs[i+1]
        n_sum += ns[i+1]
        while n_sum >= w
            excess_n = n_sum - w
            excess_x = excess_n * xs[i+1] / ns[i+1]
            res_x[trailing_i+1] = (x_sum - excess_x) / w
            x_sum -= xs[trailing_i+1]
            n_sum -= ns[trailing_i+1]
            trailing_i -= 1
        end
    end

    res_h = hs[(trailing_i + 2):end]
    res_x = res_x[(trailing_i + 2):end]

    return DataFrame(horizon=res_h, name=>res_x)
end


using DataFrames, Statistics, LinearAlgebra

function rolling_median_by_h(x, h, w, name)
    df = DataFrame(x=x, h=h)
    grouped = groupby(df, :h)
    df2 = combine(grouped, nrow)
    sort!(df2, :h)
    hs = df2.h

    res_h = []
    res_x = []
    i = length(hs) - 1
    while i >= 0
        h_i = hs[i+1]
        xs = grouped[h_i].x |> Vector

        next_idx_to_add = findfirst(==(h_i), h) - 1
        while (length(xs) < w) && (next_idx_to_add >= 1)
            push!(xs, x[next_idx_to_add])
            next_idx_to_add -= 1
        end
        if length(xs) < w
            break
        end
        push!(res_h, hs[i+1])
        push!(res_x, median(xs))
        i -= 1
    end
    reverse!(res_h)
    reverse!(res_x)
    return DataFrame(horizon=res_h, name=>res_x)
end

function mse(df, w)
    se = (df.y .- df.yhat) .^ 2
    if w < 0
        return DataFrame(horizon=df.horizon, mse=se)
    end
    return rolling_mean_by_h(se, df.horizon, w, "mse")
end

function rmse(df, w)
    res = mse(df, w)
    res.mse = sqrt.(res.mse)
    rename!(res, :mse => :rmse)
    return res
end


function mae(df, w)
    ae = abs.(df.y .- df.yhat)
    if w < 0
        return DataFrame(horizon=df.horizon, mae=ae)
    end
    return rolling_mean_by_h(ae, df.horizon, w, "mae")
end

function mape(df, w)
    ape = abs.((df.y .- df.yhat) ./ df.y)
    if w < 0
        return DataFrame(horizon=df.horizon, mape=ape)
    end
    return rolling_mean_by_h(ape, df.horizon, w, "mape")
end

function mdape(df, w)
    ape = abs.((df.y .- df.yhat) ./ df.y)
    if w < 0
        return DataFrame(horizon=df.horizon, mdape=ape)
    end
    return rolling_median_by_h(ape, df.horizon, w, "mdape")
end


function smape(df, w)
    denom = (abs.(df.y) .+ abs.(df.yhat)) ./ 2
    sape = abs.(df.y - df.yhat) ./ denom
    if w < 0
        return DataFrame(horizon = df.horizon, smape = sape)
    end
    return rolling_mean_by_h(sape, df.horizon, w, "smape")
end

function coverage(df, w)
    is_cov = (df.yhat_lower .<= df.y) .& (df.y .<= df.yhat_upper)
    if w < 0
        return DataFrame(horizon = df.horizon, coverage = is_cov)
    end
    return rolling_mean_by_h(is_cov, df.horizon, w, "coverage")
end