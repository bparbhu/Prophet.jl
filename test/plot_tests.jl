@testset "Plot backends by model backend" begin
    for backend in FAST_BACKENDS
        @testset "$(backend)" begin
            m = Prophet.ProphetModel(model_backend=backend)
            fit(m, example_daily(20))
            fcst = predict(m, make_future_dataframe(m; periods=3))

            @test model_backend(m) == backend
            @test plot_forecast(m, fcst; backend=:makie) isa CairoMakie.Figure
            @test plot_forecast(m, fcst; backend=:gadfly) isa Gadfly.Plot
            @test plot_forecast_component(m, fcst, "trend"; backend=:makie) isa CairoMakie.Figure
            @test plot_forecast_component(m, fcst, "trend"; backend=:gadfly) isa Gadfly.Plot
        end
    end
end
