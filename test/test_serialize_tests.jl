@testset "Python test_serialize.py parity by backend" begin
    @testset "unfit models cannot be serialized" begin
        @test_throws ErrorException model_to_dict(Prophet.ProphetModel(model_backend=:turing))
    end

    for backend in FAST_BACKENDS
        @testset "$(backend)" begin
            df = example_daily(35)
            df.promo = Float64.(mod.(collect(1:nrow(df)), 3))
            holidays = DataFrame(holiday=["launch"], ds=[Date(2020, 1, 10)], lower_window=[0], upper_window=[1])

            m = Prophet.ProphetModel(model_backend=backend, holidays=holidays)
            add_seasonality(m; name="monthly", period=30.5, fourier_order=3)
            add_regressor(m; name="promo", standardize=true)
            fit(m, df)

            model_dict = model_to_dict(m)
            @test model_dict["model_backend"] == String(backend)
            @test haskey(model_dict, "params")
            @test haskey(model_dict["params"], "k")
            @test haskey(model_dict["params"], "delta")
            @test haskey(model_dict["params"], "beta")
            @test !haskey(model_dict, "fit_result")

            restored = model_from_dict(model_dict)
            @test model_backend(restored) == backend
            @test fit_backend(restored) == backend
            @test restored.growth == m.growth
            @test restored.n_changepoints == m.n_changepoints
            @test restored.history !== nothing
            @test names(restored.history) == names(m.history)
            @test restored.history.ds == m.history.ds
            @test restored.params["k"] == m.params["k"]
            @test restored.params["delta"] == m.params["delta"]
            @test Set(keys(restored.seasonalities)) == Set(keys(m.seasonalities))
            @test Set(keys(restored.extra_regressors)) == Set(keys(m.extra_regressors))

            future = make_future_dataframe(m; periods=3, include_history=false)
            future.promo = fill(0.0, nrow(future))
            restored_fcst = predict(restored, future)
            original_fcst = predict(m, future)
            @test isapprox(restored_fcst.yhat, original_fcst.yhat)

            json = model_to_json(m)
            restored_json = model_from_json(json)
            @test model_backend(restored_json) == backend
            @test restored_json.history.ds == m.history.ds
            @test isapprox(predict(restored_json, future).yhat, original_fcst.yhat)
        end
    end
end
