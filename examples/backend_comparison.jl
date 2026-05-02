#!/usr/bin/env julia

using Dates
using DataFrames
using Statistics
using Prophet

function example_daily(n::Int=120)
    ds = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))
    t = collect(0:(n - 1))
    y = 10 .+ 0.18 .* t .+ 1.5 .* sin.(2pi .* t ./ 7) .+ 0.5 .* cos.(2pi .* t ./ 30)
    return DataFrame(ds=collect(ds), y=Float64.(y))
end

function fit_backend(df::DataFrame, backend::Symbol; periods::Int=30)
    model = Prophet.ProphetModel(model_backend=backend, uncertainty_samples=100)
    Prophet.add_country_holidays(model; country_name="US")
    Prophet.add_seasonality(model; name="monthly", period=30.5, fourier_order=5)
    Prophet.fit(model, df)

    future = Prophet.make_future_dataframe(model; periods=periods)
    forecast = Prophet.predict(model, future)
    forecast.backend = fill(String(backend), nrow(forecast))
    return model, forecast
end

function compare_backends(; periods::Int=30)
    df = example_daily()
    backends = (:stan, :turing, :neural_turing)
    fitted = Dict{Symbol,Any}()
    forecasts = DataFrame[]

    for backend in backends
        model, forecast = fit_backend(df, backend; periods=periods)
        fitted[backend] = model
        push!(forecasts, forecast)
    end

    combined = vcat(forecasts...; cols=:union)
    wide = unstack(combined[:, [:ds, :backend, :yhat]], :backend, :yhat)

    summary = DataFrame(
        backend=String[],
        final_yhat=Float64[],
        mean_yhat=Float64[],
        mean_abs_diff_vs_stan=Float64[],
    )

    for backend in backends
        yhat = wide[!, Symbol(String(backend))]
        stan_yhat = wide[!, :stan]
        push!(
            summary,
            (
                String(backend),
                last(yhat),
                mean(yhat),
                mean(abs.(yhat .- stan_yhat)),
            ),
        )
    end

    return fitted, combined, summary
end

function main()
    _, forecasts, summary = compare_backends()
    println("Backend comparison summary")
    show(summary, allrows=true, allcols=true)
    println()
    println()
    println("Forecast preview")
    show(last(forecasts, 9), allrows=true, allcols=true)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
