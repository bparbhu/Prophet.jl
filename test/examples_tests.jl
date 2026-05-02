@testset "Examples" begin
    include(joinpath(@__DIR__, "..", "examples", "backend_comparison.jl"))
    fitted, forecasts, summary = compare_backends(periods=5)

    @test Set(keys(fitted)) == Set(BACKENDS)
    @test nrow(summary) == length(BACKENDS)
    @test all(["backend", "final_yhat", "mean_yhat", "mean_abs_diff_vs_stan"] .in Ref(names(summary)))
    @test Set(summary.backend) == Set(String.(BACKENDS))
    @test nrow(forecasts) == length(BACKENDS) * (120 + 5)
end
