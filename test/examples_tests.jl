@testset "Examples" begin
    include(joinpath(@__DIR__, "..", "examples", "backend_comparison.jl"))
    include(joinpath(@__DIR__, "..", "examples", "stan_backend.jl"))
    include(joinpath(@__DIR__, "..", "examples", "turing_backend.jl"))
    include(joinpath(@__DIR__, "..", "examples", "neural_turing_backend.jl"))

    @test isdefined(Main, :run_stan_backend_example)
    @test isdefined(Main, :run_turing_backend_example)
    @test isdefined(Main, :run_neural_turing_backend_example)

    @testset "Standalone backend examples execute" begin
        example_runs = (
            (:stan, run_stan_backend_example(; periods=2, n=12)),
            (:turing, run_turing_backend_example(; periods=2, n=24)),
            (:neural_turing, run_neural_turing_backend_example(; periods=2, n=24)),
        )

        for (backend, result) in example_runs
            model, forecast, example_summary = result
            @test model_backend(model) == backend
            @test fit_backend(model) == backend
            @test nrow(forecast) == (backend == :stan ? 12 : 24) + 2
            @test nrow(example_summary) == 1
            @test example_summary.backend == [String(backend)]
            @test all(in.(["backend", "fit_engine", "forecast_rows", "final_yhat", "mean_yhat"], Ref(names(example_summary))))
            @test example_summary.forecast_rows[1] == nrow(forecast)
            @test all(isfinite, forecast.yhat)
        end
    end

    fitted, forecasts, summary = compare_backends(periods=5, backends=FAST_BACKENDS)

    @test Set(keys(fitted)) == Set(FAST_BACKENDS)
    @test nrow(summary) == length(FAST_BACKENDS)
    @test all(in.(["backend", "final_yhat", "mean_yhat", "mean_abs_diff_vs_reference"], Ref(names(summary))))
    @test Set(summary.backend) == Set(String.(FAST_BACKENDS))
    @test nrow(forecasts) == length(FAST_BACKENDS) * (120 + 5)
end
