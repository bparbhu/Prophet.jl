@testset "Examples" begin
    include(joinpath(@__DIR__, "..", "examples", "backend_comparison.jl"))
    fitted, forecasts, summary = compare_backends(periods=5, backends=FAST_BACKENDS)

    @test Set(keys(fitted)) == Set(FAST_BACKENDS)
    @test nrow(summary) == length(FAST_BACKENDS)
    @test all(in.(["backend", "final_yhat", "mean_yhat", "mean_abs_diff_vs_reference"], Ref(names(summary))))
    @test Set(summary.backend) == Set(String.(FAST_BACKENDS))
    @test nrow(forecasts) == length(FAST_BACKENDS) * (120 + 5)
end
