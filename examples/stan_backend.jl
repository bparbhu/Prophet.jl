#!/usr/bin/env julia

using Dates
using DataFrames
using Statistics
using Prophet

function stan_example_daily(n::Int=90)
    ds = Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(n - 1))
    t = collect(0:(n - 1))
    y = 25 .+ 0.12 .* t .+ 1.8 .* sin.(2pi .* t ./ 7)
    return DataFrame(ds=collect(ds), y=Float64.(y))
end

function run_stan_backend_example(; periods::Int=14, n::Int=90)
    df = stan_example_daily(n)
    model = Prophet.ProphetModel(
        model_backend=:stan,
        weekly_seasonality=true,
        yearly_seasonality=false,
        daily_seasonality=false,
    )
    Prophet.add_country_holidays(model, "US")
    Prophet.fit(model, df)

    future = Prophet.make_future_dataframe(model; periods=periods)
    forecast = Prophet.predict(model, future)
    summary = DataFrame(
        backend=[String(Prophet.fit_backend(model))],
        fit_engine=[String(Prophet.fit_engine(model))],
        forecast_rows=[nrow(forecast)],
        final_yhat=[last(forecast.yhat)],
        mean_yhat=[mean(forecast.yhat)],
    )
    return model, forecast, summary
end

function main()
    _, forecast, summary = run_stan_backend_example()
    println("Stan backend example")
    show(summary, allrows=true, allcols=true)
    println()
    println()
    show(last(forecast, 6), allrows=true, allcols=true)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
