@testset "Python test_diagnostics.py parity by backend" begin
    df = example_daily(42)

    function sorted_cv(df)
        out = copy(df)
        sort!(out, [:cutoff, :ds])
        return out
    end

    function assert_parallel_cv_matches(reference, candidate)
        reference = sorted_cv(reference)
        candidate = sorted_cv(candidate)
        @test names(candidate) == names(reference)
        @test nrow(candidate) == nrow(reference)
        @test candidate.ds == reference.ds
        @test candidate.cutoff == reference.cutoff
        @test candidate.y == reference.y
        @test candidate.model_backend == reference.model_backend
        @test all(isfinite, candidate.yhat)
        if "yhat_lower" in names(reference)
            @test all(isfinite, candidate.yhat_lower)
            @test all(isfinite, candidate.yhat_upper)
            @test all(candidate.yhat_lower .<= candidate.yhat_upper)
        end
    end

    @testset "cross validation cutoff contract" begin
        cutoffs = generate_cutoffs(df, Day(4), Day(20), Day(7))
        @test !isempty(cutoffs)
        @test length(unique(cutoffs)) == length(cutoffs)
        @test maximum(df.ds) - maximum(cutoffs) == Day(4)
        @test minimum(cutoffs) >= minimum(df.ds) + Day(20)
        @test all(cutoffs .< maximum(df.ds))
    end

    @testset "cross validation by backend and parallel mode" begin
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                m = Prophet.ProphetModel(model_backend=backend)
                fit(m, df)

                cv = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20))
                cv_threads = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20), parallel=:threads)
                cv_dagger = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20), parallel=:dagger)
                cv_dask_alias = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20), parallel="dask")
                cv_threads_alias = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20), parallel="threads")
                cv_processes_alias = cross_validation(m; horizon=Day(4), period=Day(7), initial=Day(20), parallel=:processes)

                @test nrow(cv) == nrow(cv_threads) == nrow(cv_dagger) == nrow(cv_dask_alias)
                assert_parallel_cv_matches(cv, cv_threads)
                assert_parallel_cv_matches(cv, cv_dagger)
                assert_parallel_cv_matches(cv, cv_dask_alias)
                assert_parallel_cv_matches(cv, cv_threads_alias)
                assert_parallel_cv_matches(cv, cv_processes_alias)
                @test all(Date.(cv.cutoff) .< Date.(cv.ds))
                @test maximum(Date.(cv.ds) .- Date.(cv.cutoff)) == Day(4)
                merged = innerjoin(cv[:, [:ds, :y]], df, on=:ds, makeunique=true)
                @test all(isapprox.(merged.y, merged.y_1))
                cv_trend = cross_validation(
                    m; horizon=Day(2), period=Day(7), initial=Day(20), extra_output_columns="trend",
                )
                @test "trend" in names(cv_trend)
                @test_throws ErrorException cross_validation(m; horizon=Day(4), parallel=:bad)
            end
        end
    end

    @testset "cross validation with logistic, flat, regressors, and conditions" begin
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                flat_model = Prophet.ProphetModel(growth="flat", model_backend=backend)
                fit(flat_model, df)
                flat_cv = cross_validation(flat_model; horizon=Day(2), period=Day(7), initial=Day(20))
                @test all(Date.(flat_cv.cutoff) .< Date.(flat_cv.ds))

                logistic_df = copy(df)
                logistic_df.cap = fill(100.0, nrow(logistic_df))
                logistic_model = Prophet.ProphetModel(growth="logistic", model_backend=backend)
                fit(logistic_model, logistic_df)
                logistic_cv = cross_validation(logistic_model; horizon=Day(2), period=Day(7), initial=Day(20))
                @test all(Date.(logistic_cv.cutoff) .< Date.(logistic_cv.ds))

                regressor_df = copy(df)
                regressor_df.extra = collect(1:nrow(regressor_df))
                regressor_df.is_conditional_week = Bool.(mod.(collect(1:nrow(regressor_df)), 2))
                regressor_model = Prophet.ProphetModel(model_backend=backend)
                add_seasonality(
                    regressor_model;
                    name="conditional_weekly",
                    period=7,
                    fourier_order=3,
                    prior_scale=2.0,
                    condition_name="is_conditional_week",
                )
                add_regressor(regressor_model; name="extra")
                fit(regressor_model, regressor_df)
                regressor_cv = cross_validation(regressor_model; horizon=Day(2), period=Day(7), initial=Day(20))
                @test all(Date.(regressor_cv.cutoff) .< Date.(regressor_cv.ds))
            end
        end
    end

    @testset "performance metrics contract" begin
        metric_df = DataFrame(
            ds=[Date(2020, 1, 3), Date(2020, 1, 4), Date(2020, 1, 5)],
            cutoff=[Date(2020, 1, 1), Date(2020, 1, 1), Date(2020, 1, 1)],
            y=[1.0, 2.0, 4.0],
            yhat=[1.5, 1.5, 5.0],
            yhat_lower=[1.0, 1.0, 4.0],
            yhat_upper=[2.0, 2.0, 6.0],
        )
        raw = performance_metrics(metric_df; metrics=[:mse], rolling_window=-1)
        @test nrow(raw) == nrow(metric_df)
        metrics = performance_metrics(metric_df; metrics=[:mse, :rmse, :mae, :mape, :mdape, :smape, :coverage])
        @test all(in.(["horizon", "mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"], Ref(names(metrics))))
        @test all(metrics.mse .>= 0)
        @test all((0 .<= metrics.coverage) .& (metrics.coverage .<= 1))
    end
end
