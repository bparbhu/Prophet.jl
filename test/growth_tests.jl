@testset "Growth helpers based on Python Prophet tests by backend" begin
    t = collect(0.0:10.0)
    y = piecewise_linear(t, [0.5], 1.0, 0.0, [5.0])
    @test isapprox(y, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5])

    cap = fill(10.0, length(t))
    y_logistic = piecewise_logistic(t, cap, [0.5], 1.0, 0.0, [5.0])
    @test isapprox(y_logistic[1], 5.0; atol=1e-6)
    @test y_logistic[end] > 9.99

    @test flat_trend(t, 0.5) == fill(0.5, length(t))

    for backend in BACKENDS
        @testset "$(backend)" begin
            flat_model = Prophet.ProphetModel(growth="flat", model_backend=backend)
            df = example_daily(20)
            fit(flat_model, df)
            future = make_future_dataframe(flat_model; periods=5, include_history=true)
            fcst = predict(flat_model, future)
            @test model_backend(flat_model) == backend
            @test length(unique(round.(fcst.trend; digits=8))) == 1

            logistic_df = example_daily(20)
            logistic_df.floor = fill(1.0, nrow(logistic_df))
            logistic_df.cap = fill(100.0, nrow(logistic_df))
            logistic_model = Prophet.ProphetModel(growth="logistic", model_backend=backend)
            hist = setup_dataframe(logistic_model, logistic_df; initialize_scales=true)
            @test model_backend(logistic_model) == backend
            @test logistic_model.logistic_floor
            @test "cap_scaled" in names(hist)
            @test all(hist.cap_scaled .> 0)
        end
    end
end
