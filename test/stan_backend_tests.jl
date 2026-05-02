@testset "Stan backend artifact" begin
    @test isfile(stan_model_file())
    @test stan_backend_module() === Stan
    stan_source = read(stan_model_file(), String)
    @test occursin("normal_id_glm", stan_source)
    @test occursin("logistic_trend", stan_source)

    if get(ENV, "CI", "false") == "true"
        @test cmdstan_version() == REQUIRED_CMDSTAN_VERSION
    elseif haskey(ENV, "CMDSTAN") || haskey(ENV, "JULIA_CMDSTAN_HOME")
        @test cmdstan_version().major == REQUIRED_CMDSTAN_VERSION.major
    end
end
