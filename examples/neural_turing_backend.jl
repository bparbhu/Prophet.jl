#!/usr/bin/env julia

using Dates
using DataFrames
using Statistics
using Prophet

function neural_turing_example_daily(n::Int=90)
    ds = Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(n - 1))
    t = collect(0:(n - 1))
    event = Float64.(mod.(t, 21) .== 0)
    nonlinear = 0.6 .* sin.(2pi .* t ./ 30) .* cos.(2pi .* t ./ 7)
    y = 12 .+ 0.1 .* t .+ 1.4 .* sin.(2pi .* t ./ 7) .+ nonlinear .+ event
    return DataFrame(ds=collect(ds), y=Float64.(y), event=event)
end

function run_neural_turing_backend_example(; periods::Int=14, n::Int=90)
    df = neural_turing_example_daily(n)
    model = Prophet.ProphetModel(
        model_backend=:neural_turing,
        weekly_seasonality=true,
        yearly_seasonality=false,
        daily_seasonality=false,
        n_changepoints=8,
    )
    Prophet.add_regressor(model, "event"; standardize=false, mode="additive")
    Prophet.fit(model, df)

    future = Prophet.make_future_dataframe(model; periods=periods)
    future.event = vcat(df.event, zeros(periods))
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
    _, forecast, summary = run_neural_turing_backend_example()
    println("Neural Turing backend example")
    show(summary, allrows=true, allcols=true)
    println()
    println()
    show(last(forecast, 6), allrows=true, allcols=true)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
