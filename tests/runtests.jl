using Test
using Dates
using DataFrames
import CairoMakie
import Gadfly
import Dagger
using Prophet

function example_daily(n::Int=120)
    ds = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))
    y = 10 .+ 0.25 .* collect(0:(n - 1)) .+ sin.(2pi .* collect(0:(n - 1)) ./ 7)
    return DataFrame(ds=collect(ds), y=Float64.(y))
end

@testset "Prophet-style constructor and validation" begin
    m = Prophet.Prophet()
    @test m.growth == "linear"
    @test m.n_changepoints == 25
    @test m.seasonality_mode == "additive"
    @test m.holidays_mode == "additive"

    @test_throws ErrorException Prophet.Prophet(growth="constant")
    @test_throws ErrorException Prophet.Prophet(changepoint_range=-0.1)
    @test_throws ErrorException Prophet.Prophet(changepoint_range=2.0)
    @test_throws ErrorException Prophet.Prophet(seasonality_mode="bad")
end

@testset "Data prep, fit, predict, and future frames" begin
    df = example_daily(30)
    m = Prophet.Prophet()
    history = setup_dataframe(m, df; initialize_scales=true)

    @test "t" in names(history)
    @test "y_scaled" in names(history)
    @test minimum(history.t) == 0.0
    @test maximum(history.t) == 1.0
    @test maximum(abs.(history.y_scaled)) <= 1.0 + eps()

    @test fit(m, df) === m
    @test m.history !== nothing
    @test m.params["k"] > 0

    future = make_future_dataframe(m; periods=3, include_history=false)
    @test nrow(future) == 3
    @test future.ds == [Date(2020, 1, 31), Date(2020, 2, 1), Date(2020, 2, 2)]

    forecast = predict(m, future)
    @test names(forecast) == ["ds", "trend", "yhat", "yhat_lower", "yhat_upper", "trend_lower", "trend_upper"]
    @test nrow(forecast) == 3
    @test all(forecast.yhat_lower .<= forecast.yhat)
    @test all(forecast.yhat .<= forecast.yhat_upper)

    no_uncertainty = Prophet.Prophet(uncertainty_samples=0)
    fit(no_uncertainty, df)
    fcst = predict(no_uncertainty, future)
    @test names(fcst) == ["ds", "trend", "yhat"]
end

@testset "Growth helpers based on Python Prophet tests" begin
    t = collect(0.0:10.0)
    y = piecewise_linear(t, [0.5], 1.0, 0.0, [5.0])
    @test isapprox(y, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5])

    cap = fill(10.0, length(t))
    y_logistic = piecewise_logistic(t, cap, [0.5], 1.0, 0.0, [5.0])
    @test isapprox(y_logistic[1], 5.0; atol=1e-6)
    @test y_logistic[end] > 9.99

    @test flat_trend(t, 0.5) == fill(0.5, length(t))

    flat_model = Prophet.Prophet(growth="flat")
    df = example_daily(20)
    fit(flat_model, df)
    future = make_future_dataframe(flat_model; periods=5, include_history=true)
    fcst = predict(flat_model, future)
    @test length(unique(round.(fcst.trend; digits=8))) == 1

    logistic_df = example_daily(20)
    logistic_df.floor = fill(1.0, nrow(logistic_df))
    logistic_df.cap = fill(100.0, nrow(logistic_df))
    logistic_model = Prophet.Prophet(growth="logistic")
    hist = setup_dataframe(logistic_model, logistic_df; initialize_scales=true)
    @test logistic_model.logistic_floor
    @test "cap_scaled" in names(hist)
    @test all(hist.cap_scaled .> 0)
end

@testset "Seasonality and regressor API" begin
    m = Prophet.Prophet()
    add_seasonality(m; name="monthly", period=30.5, fourier_order=3)
    @test haskey(m.seasonalities, "monthly")
    @test m.seasonalities["monthly"]["fourier_order"] == 3
    @test_throws ErrorException add_seasonality(m; name="bad", period=7, fourier_order=0)

    add_regressor(m; name="promo", prior_scale=2.0, standardize=false, mode="multiplicative")
    @test haskey(m.extra_regressors, "promo")
    @test m.extra_regressors["promo"]["mode"] == "multiplicative"

    dates = [Date(2012, 6, 1)]
    weekly = fourier_series(dates, 7, 3)
    @test size(weekly) == (1, 6)
    @test isapprox(
        weekly[1, :],
        [0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689];
        atol=1e-6,
    )

    features = make_seasonality_features(dates, 7, 3, "weekly")
    @test names(features) == ["weekly_delim_1", "weekly_delim_2", "weekly_delim_3", "weekly_delim_4", "weekly_delim_5", "weekly_delim_6"]
end

@testset "Embedded holidays and holiday features" begin
    countries = supported_holiday_countries()
    @test "US" in countries

    us = make_holidays_df([2015], "US")
    @test names(us) == ["ds", "holiday"]
    @test Date(2015, 1, 1) in us.ds
    @test "New Year's Day" in get_holiday_names("US")
    @test_throws ErrorException make_holidays_df([2015], "NO_SUCH_COUNTRY")

    m = Prophet.Prophet(holidays=DataFrame(
        holiday=["launch"],
        ds=[Date(2020, 1, 10)],
        lower_window=[-1],
        upper_window=[1],
        prior_scale=[4.0],
    ))
    holidays = construct_holiday_dataframe(m, Date(2020, 1, 8):Day(1):Date(2020, 1, 12))
    features, prior_scales, names_used = make_holiday_features(
        m, Date(2020, 1, 8):Day(1):Date(2020, 1, 12), holidays
    )
    @test names(features) == ["launch_delim_+0", "launch_delim_+1", "launch_delim_-1"]
    @test prior_scales == [4.0, 4.0, 4.0]
    @test names_used == ["launch"]
    @test sum(Matrix(features)) == 3.0

    m2 = Prophet.Prophet()
    add_country_holidays(m2; country_name="US")
    @test m2.country_holidays == "US"
end

@testset "Stan-equivalent Turing utilities" begin
    t = [0.0, 0.5, 1.0]
    t_change = [0.25, 0.75]
    A = get_changepoint_matrix(t, t_change)
    @test A == [0.0 0.0; 1.0 0.0; 1.0 1.0]

    delta = [0.1, -0.05]
    @test isapprox(
        linear_trend(1.0, 0.2, delta, t, A, t_change),
        (1.0 .+ A * delta) .* t .+ (0.2 .+ A * (-t_change .* delta)),
    )

    logistic = logistic_trend(1.0, 0.1, delta, t, ones(3), A, t_change)
    @test all((0 .< logistic) .& (logistic .< 1))

    X = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    beta = [0.2, -0.1]
    s_a = [1.0, 0.0]
    s_m = [0.0, 1.0]
    base_mu = prophet_mean(1.0, 0.2, delta, beta, t, ones(3), t_change, X, 0, s_a, s_m)
    expected_trend = linear_trend(1.0, 0.2, delta, t, A, t_change)
    expected_mu = (X .* reshape(s_a, 1, :)) * beta .+
                  expected_trend .* (1 .+ (X .* reshape(s_m, 1, :)) * beta)
    @test isapprox(base_mu, expected_mu)

    flat_mu = prophet_mean(1.0, 2.5, Float64[], beta, t, zeros(3), Float64[], X, 2, s_a, s_m)
    @test isapprox(
        flat_mu,
        (X .* reshape(s_a, 1, :)) * beta .+
        fill(2.5, 3) .* (1 .+ (X .* reshape(s_m, 1, :)) * beta),
    )

    model = prophet(
        3, 1, [0.0, 0.5, 1.0], zeros(3), [0.1, 0.2, 0.3], 1, [0.5],
        reshape([1.0, 0.0, 1.0], 3, 1), [1.0], 0.05, 0, [1.0], [0.0],
    )
    @test model !== nothing
end

@testset "Flux-Turing neural extension" begin
    t = [0.0, 0.5, 1.0]
    t_change = [0.25]
    delta = [0.1]
    X = reshape([1.0, 0.0, 1.0], 3, 1)
    beta = [0.2]
    s_a = [1.0]
    s_m = [0.0]
    X_seasonality = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    X_autoregression = reshape([0.5, 0.25, 0.0], 3, 1)

    base_mu = prophet_mean(1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m)
    zero_neural_mu = neural_prophet_mean(
        1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m,
        X_seasonality, X_autoregression, zeros(2), zeros(1),
    )
    @test isapprox(zero_neural_mu, base_mu)

    W_seasonality = [0.1, -0.2]
    W_autoregression = [0.5]
    neural_mu = neural_prophet_mean(
        1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m,
        X_seasonality, X_autoregression, W_seasonality, W_autoregression,
    )
    @test isapprox(
        neural_mu,
        base_mu .+ X_seasonality * W_seasonality .+ X_autoregression * W_autoregression,
    )

    nn = NeuralProphetNN(W_seasonality, W_autoregression)
    @test isapprox(
        nn(X_seasonality, X_autoregression),
        X_seasonality * W_seasonality .+ X_autoregression * W_autoregression,
    )

    nn_model = neural_prophet(
        3, 1, [0.0, 0.5, 1.0], zeros(3), [0.1, 0.2, 0.3], 1, [0.5],
        reshape([1.0, 0.0, 1.0], 3, 1), [1.0], 0.05, 0, [1.0], [0.0],
        ones(3, 2), zeros(3, 1),
    )
    @test nn_model !== nothing
end

@testset "Diagnostics based on Python Prophet tests" begin
    df = example_daily(30)
    m = Prophet.Prophet()
    fit(m, df)

    cutoffs = generate_cutoffs(df, Day(4), Day(12), Day(4))
    @test !isempty(cutoffs)
    @test all(cutoffs .< maximum(df.ds))

    cv = cross_validation(m; horizon=Day(4), initial=Day(12), period=Day(4))
    @test all(["ds", "yhat", "y", "cutoff"] .in Ref(names(cv)))
    @test all(Date.(cv.ds) .> Date.(cv.cutoff))
    @test all(Date.(cv.ds) .<= Date.(cv.cutoff) .+ Day(4))

    cv_threads = cross_validation(m; horizon=Day(4), initial=Day(12), period=Day(4), parallel=:threads)
    @test nrow(cv_threads) == nrow(cv)

    metrics = performance_metrics(cv; metrics=[:mse, :rmse, :mae, :mape, :smape, :coverage])
    @test all(["horizon", "mse", "rmse", "mae", "mape", "smape", "coverage"] .in Ref(names(metrics)))
    @test all(metrics.mse .>= 0)
    @test all(metrics.rmse .>= 0)
    @test all((0 .<= metrics.coverage) .& (metrics.coverage .<= 1))

    raw_metrics = performance_metrics(cv; metrics=[:mse], rolling_window=-1)
    @test nrow(raw_metrics) == nrow(cv)
end

@testset "Plot backends" begin
    m = Prophet.Prophet()
    fit(m, example_daily(20))
    fcst = predict(m, make_future_dataframe(m; periods=3))

    @test plot_forecast(m, fcst; backend=:makie) isa CairoMakie.Figure
    @test plot_forecast(m, fcst; backend=:gadfly) isa Gadfly.Plot
    @test plot_forecast_component(m, fcst, "trend"; backend=:makie) isa CairoMakie.Figure
    @test plot_forecast_component(m, fcst, "trend"; backend=:gadfly) isa Gadfly.Plot
end
