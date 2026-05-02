@testset "Data prep, fit, predict, and future frames by backend" begin
    for backend in FAST_BACKENDS
        @testset "$(backend)" begin
            df = example_daily(30)
            m = Prophet.ProphetModel(model_backend=backend)
            history = setup_dataframe(m, df; initialize_scales=true)

            @test model_backend(m) == backend
            @test "t" in names(history)
            @test "y_scaled" in names(history)
            @test minimum(history.t) == 0.0
            @test maximum(history.t) == 1.0
            @test maximum(abs.(history.y_scaled)) <= 1.0 + eps()

            @test fit(m, df) === m
            @test model_backend(m) == backend
            @test fit_backend(m) == backend
            @test fit_engine(m) == expected_fit_engine(backend)
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

            no_uncertainty = Prophet.ProphetModel(model_backend=backend, uncertainty_samples=0)
            fit(no_uncertainty, df)
            fcst = predict(no_uncertainty, future)
            @test model_backend(no_uncertainty) == backend
            @test fit_backend(no_uncertainty) == backend
            @test fit_engine(no_uncertainty) == expected_fit_engine(backend)
            @test names(fcst) == ["ds", "trend", "yhat"]
        end
    end

    @testset "stan" begin
        df = example_daily(12)
        m = Prophet.ProphetModel(model_backend=:stan, n_changepoints=2)
        @test fit(m, df) === m
        @test fit_backend(m) == :stan
        @test fit_engine(m) == :stan_optimize
        forecast = predict(m, make_future_dataframe(m; periods=2, include_history=false))
        @test nrow(forecast) == 2
    end
end
