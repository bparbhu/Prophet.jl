@testset "Python test_prophet.py parity by backend" begin
    @testset "fit/predict edge cases" begin
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                df = example_daily(40)

                no_seasons = Prophet.ProphetModel(
                    model_backend=backend,
                    weekly_seasonality=false,
                    yearly_seasonality=false,
                    daily_seasonality=false,
                )
                fit(no_seasons, first(df, 25))
                future = make_future_dataframe(no_seasons; periods=5, include_history=false)
                result = predict(no_seasons, future)
                @test result.ds == future.ds

                no_changepoints = Prophet.ProphetModel(model_backend=backend, n_changepoints=0)
                fit(no_changepoints, first(df, 25))
                @test no_changepoints.n_changepoints == 0
                @test length(no_changepoints.params["delta"]) == 0
                @test predict(no_changepoints, last(df, 5)[:, [:ds]]) isa DataFrame

                repeated = copy(first(df, 20))
                repeated.y .+= 10
                duplicated = vcat(first(df, 20), repeated)
                duplicate_model = Prophet.ProphetModel(model_backend=backend)
                fit(duplicate_model, duplicated)
                @test predict(duplicate_model, last(df, 5)[:, [:ds]]) isa DataFrame

                for constant in (0.0, 20.0)
                    constant_df = copy(first(df, 25))
                    constant_df.y .= constant
                    constant_model = Prophet.ProphetModel(model_backend=backend)
                    fit(constant_model, constant_df)
                    constant_fcst = predict(constant_model, last(df, 5)[:, [:ds]])
                    @test all(isapprox.(constant_fcst.yhat, constant; atol=1e-3))
                end

                no_uncertainty = Prophet.ProphetModel(model_backend=backend, uncertainty_samples=0)
                fit(no_uncertainty, first(df, 25))
                no_uncertainty_fcst = predict(no_uncertainty, last(df, 5)[:, [:ds]])
                @test names(no_uncertainty_fcst) == ["ds", "trend", "yhat"]
            end
        end
    end

    @testset "data prep and future frames" begin
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                df = example_daily(30)
                m = Prophet.ProphetModel(model_backend=backend)
                history = setup_dataframe(m, df; initialize_scales=true)
                @test "t" in names(history)
                @test "y_scaled" in names(history)
                @test minimum(history.t) == 0.0
                @test maximum(history.t) == 1.0
                @test maximum(abs.(history.y_scaled)) <= 1.0 + eps()

                missing_df = DataFrame(ds=df.ds, y=Vector{Union{Missing,Float64}}(df.y))
                missing_df.y[3] = missing
                missing_model = Prophet.ProphetModel(model_backend=backend)
                fit(missing_model, missing_df)
                future = make_future_dataframe(missing_model; periods=3, include_history=true)
                @test nrow(future) == nrow(missing_df) + 3

                logistic_df = copy(df)
                logistic_df.floor = fill(1.0, nrow(logistic_df))
                logistic_df.cap = fill(100.0, nrow(logistic_df))
                logistic_model = Prophet.ProphetModel(growth="logistic", model_backend=backend)
                fit(logistic_model, logistic_df)
                @test logistic_model.logistic_floor
                @test "floor" in names(logistic_model.history)
                @test "cap_scaled" in names(logistic_model.history)
            end
        end
    end

    @testset "trend helpers and changepoints" begin
        df = example_daily(50)
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                m = Prophet.ProphetModel(model_backend=backend, changepoint_range=0.4)
                history = setup_dataframe(m, df; initialize_scales=true)
                cps = Prophet._changepoints_t(m, history)
                @test length(cps) == m.n_changepoints
                @test minimum(cps) > 0
                cp_index = ceil(Int, 0.4 * nrow(history))
                @test maximum(cps) <= history.t[cp_index]

                zero_cp = Prophet.ProphetModel(model_backend=backend, n_changepoints=0)
                zero_history = setup_dataframe(zero_cp, df; initialize_scales=true)
                @test isempty(Prophet._changepoints_t(zero_cp, zero_history))
            end
        end
    end
end
