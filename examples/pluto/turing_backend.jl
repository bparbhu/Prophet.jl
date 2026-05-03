### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-aaaa-1111-1111-111111111111
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 22222222-aaaa-2222-2222-222222222222
begin
    using Dates
    using DataFrames
    using Statistics
    using Prophet
end

# ╔═╡ 33333333-aaaa-3333-3333-333333333333
md"""
# Turing Backend

This notebook fits the Stan-equivalent Prophet model implemented in Turing.
"""

# ╔═╡ 44444444-aaaa-4444-4444-444444444444
begin
    n = 90
    ds = collect(Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(n - 1)))
    t = collect(0:(n - 1))
    promo = Float64.(mod.(t, 14) .< 4)
    y = 18 .+ 0.08 .* t .+ 1.25 .* sin.(2pi .* t ./ 7) .+ 0.75 .* promo
    df = DataFrame(ds=ds, y=Float64.(y), promo=promo)
end

# ╔═╡ 55555555-aaaa-5555-5555-555555555555
begin
    model = Prophet.ProphetModel(
        model_backend=:turing,
        weekly_seasonality=true,
        yearly_seasonality=false,
        daily_seasonality=false,
        n_changepoints=8,
    )
    Prophet.add_regressor(model, "promo"; standardize=false, mode="additive")
    Prophet.fit(model, df)
end

# ╔═╡ 66666666-aaaa-6666-6666-666666666666
begin
    future = Prophet.make_future_dataframe(model; periods=14)
    future.promo = vcat(df.promo, zeros(14))
    forecast = Prophet.predict(model, future)
end

# ╔═╡ 77777777-aaaa-7777-7777-777777777777
DataFrame(
    backend=[String(Prophet.fit_backend(model))],
    fit_engine=[String(Prophet.fit_engine(model))],
    forecast_rows=[nrow(forecast)],
    final_yhat=[last(forecast.yhat)],
    mean_yhat=[mean(forecast.yhat)],
)

# ╔═╡ 88888888-aaaa-8888-8888-888888888888
last(forecast, 10)

# ╔═╡ Cell order:
# ╠═11111111-aaaa-1111-1111-111111111111
# ╠═22222222-aaaa-2222-2222-222222222222
# ╟─33333333-aaaa-3333-3333-333333333333
# ╠═44444444-aaaa-4444-4444-444444444444
# ╠═55555555-aaaa-5555-5555-555555555555
# ╠═66666666-aaaa-6666-6666-666666666666
# ╠═77777777-aaaa-7777-7777-777777777777
# ╠═88888888-aaaa-8888-8888-888888888888
