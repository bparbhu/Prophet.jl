@testset "Stan-equivalent Turing utilities" begin
    t = [0.0, 0.5, 1.0]
    t_change = [0.25, 0.75]
    A = get_changepoint_matrix(t, t_change)
    @test A == [0.0 0.0; 1.0 0.0; 1.0 1.0]

    delta = [0.1, -0.05]
    @test isapprox(
        linear_trend(1.0, 0.2, delta, t, A, t_change),
        (1.0 .+ A * delta) .* t .+ (0.2 .+ A * (-t_change .* delta)),
    )

    logistic = logistic_trend(1.0, 0.1, delta, t, ones(3), A, t_change)
    @test all((0 .< logistic) .& (logistic .< 1))

    X = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    beta = [0.2, -0.1]
    s_a = [1.0, 0.0]
    s_m = [0.0, 1.0]
    base_mu = prophet_mean(1.0, 0.2, delta, beta, t, ones(3), t_change, X, 0, s_a, s_m)
    expected_trend = linear_trend(1.0, 0.2, delta, t, A, t_change)
    expected_mu = (X .* reshape(s_a, 1, :)) * beta .+
                  expected_trend .* (1 .+ (X .* reshape(s_m, 1, :)) * beta)
    @test isapprox(base_mu, expected_mu)

    flat_mu = prophet_mean(1.0, 2.5, Float64[], beta, t, zeros(3), Float64[], X, 2, s_a, s_m)
    @test isapprox(
        flat_mu,
        (X .* reshape(s_a, 1, :)) * beta .+
        fill(2.5, 3) .* (1 .+ (X .* reshape(s_m, 1, :)) * beta),
    )

    model = prophet(
        3, 1, [0.0, 0.5, 1.0], zeros(3), [0.1, 0.2, 0.3], 1, [0.5],
        reshape([1.0, 0.0, 1.0], 3, 1), [1.0], 0.05, 0, [1.0], [0.0],
    )
    @test model !== nothing
end
