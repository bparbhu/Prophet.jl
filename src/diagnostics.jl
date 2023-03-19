using DataFrames, Logging, Dates, Statistics, ProgressMeter, Random

logger = getlogger("prophet")

function generate_cutoffs(df, horizon, initial, period)
    cutoff = maximum(df.ds) - horizon
    if cutoff < minimum(df.ds)
        error("Less data than horizon.")
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
        error("Less data than horizon after initial window. Make horizon or initial shorter.")
    end
    @info "Making $(length(result)) forecasts with cutoffs between $(last(result)) and $(first(result))"
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

function single_cutoff_forecast(data, horizon, fold_size, model, args)
    # Determine the number of folds
    n_folds = trunc(Int, (data.date[end] - data.date[1]) / fold_size) - horizon + 1

    # Generate forecasts for each fold
    forecasts = []
    for i in 1:n_folds
        cutoff = data.date[1] + (i - 1) * fold_size
        df_train = data[data.date .<= cutoff, :]
        df_test = data[(data.date .> cutoff) .& (data.date .<= cutoff + horizon), :]

        # Fit the model and generate forecasts
        fitted_model = model(df_train, args)
        forecast = fitted_model(df_test.date)

        # Store the forecast
        push!(forecasts, forecast)
    end

    # Combine the forecasts
    df_forecast = vcat(forecasts...)

    # Calculate forecast errors
    df_errors = data[(data.date .> data.date[1] + horizon - 1) .& (data.date .<= data.date[end]), :]
    df_errors = innerjoin(df_errors, df_forecast, on = :date)
    df_errors.smape = smape(df_errors, -1).smape
    df_errors.coverage = coverage(df_errors, -1).coverage
    return df_errors
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