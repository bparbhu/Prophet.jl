@testset "Diagnostics based on Python Prophet tests by backend" begin
    df = example_daily(30)

    cutoffs = generate_cutoffs(df, Day(4), Day(12), Day(4))
    @test !isempty(cutoffs)
    @test all(cutoffs .< maximum(df.ds))

    for backend in BACKENDS
        @testset "$(backend)" begin
            m = Prophet.ProphetModel(model_backend=backend)
            fit(m, df)
            @test fit_backend(m) == backend
            @test fit_engine(m) == expected_fit_engine(backend)

            cv = cross_validation(m; horizon=Day(4), initial=Day(12), period=Day(4))
            @test model_backend(m) == backend
            @test all(in.(["ds", "yhat", "y", "cutoff", "model_backend"], Ref(names(cv))))
            @test all(cv.model_backend .== String(backend))
            @test all(Date.(cv.ds) .> Date.(cv.cutoff))
            @test all(Date.(cv.ds) .<= Date.(cv.cutoff) .+ Day(4))

            cv_threads = cross_validation(m; horizon=Day(4), initial=Day(12), period=Day(4), parallel=:threads)
            @test nrow(cv_threads) == nrow(cv)

            metrics = performance_metrics(cv; metrics=[:mse, :rmse, :mae, :mape, :smape, :coverage])
            @test all(in.(["horizon", "mse", "rmse", "mae", "mape", "smape", "coverage"], Ref(names(metrics))))
            @test all(metrics.mse .>= 0)
            @test all(metrics.rmse .>= 0)
            @test all((0 .<= metrics.coverage) .& (metrics.coverage .<= 1))

            raw_metrics = performance_metrics(cv; metrics=[:mse], rolling_window=-1)
            @test nrow(raw_metrics) == nrow(cv)
        end
    end
end
