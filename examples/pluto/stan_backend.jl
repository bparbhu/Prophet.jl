### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1111-1111-1111-111111111111
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 22222222-2222-2222-2222-222222222222
begin
    using Dates
    using DataFrames
    using Statistics
    using Prophet
end

# ╔═╡ 33333333-3333-3333-3333-333333333333
md"""
# Stan Backend

This notebook fits a Prophet-style model with the bundled Stan model through CmdStan.
"""

# ╔═╡ 44444444-4444-4444-4444-444444444444
begin
    n = 90
    ds = collect(Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(n - 1)))
    t = collect(0:(n - 1))
    y = 25 .+ 0.12 .* t .+ 1.8 .* sin.(2pi .* t ./ 7)
    df = DataFrame(ds=ds, y=Float64.(y))
end

# ╔═╡ 55555555-5555-5555-5555-555555555555
begin
    model = Prophet.ProphetModel(
        model_backend=:stan,
        weekly_seasonality=true,
        yearly_seasonality=false,
        daily_seasonality=false,
    )
    Prophet.add_country_holidays(model, "US")
    Prophet.fit(model, df)
end

# ╔═╡ 66666666-6666-6666-6666-666666666666
begin
    future = Prophet.make_future_dataframe(model; periods=14)
    forecast = Prophet.predict(model, future)
end

# ╔═╡ 77777777-7777-7777-7777-777777777777
DataFrame(
    backend=[String(Prophet.fit_backend(model))],
    fit_engine=[String(Prophet.fit_engine(model))],
    forecast_rows=[nrow(forecast)],
    final_yhat=[last(forecast.yhat)],
    mean_yhat=[mean(forecast.yhat)],
)

# ╔═╡ 88888888-8888-8888-8888-888888888888
last(forecast, 10)

# ╔═╡ Cell order:
# ╠═11111111-1111-1111-1111-111111111111
# ╠═22222222-2222-2222-2222-222222222222
# ╟─33333333-3333-3333-3333-333333333333
# ╠═44444444-4444-4444-4444-444444444444
# ╠═55555555-5555-5555-5555-555555555555
# ╠═66666666-6666-6666-6666-666666666666
# ╠═77777777-7777-7777-7777-777777777777
# ╠═88888888-8888-8888-8888-888888888888
