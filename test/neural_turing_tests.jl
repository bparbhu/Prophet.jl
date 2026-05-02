@testset "Flux-Turing neural extension" begin
    t = [0.0, 0.5, 1.0]
    t_change = [0.25]
    delta = [0.1]
    X = reshape([1.0, 0.0, 1.0], 3, 1)
    beta = [0.2]
    s_a = [1.0]
    s_m = [0.0]
    X_seasonality = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    X_autoregression = reshape([0.5, 0.25, 0.0], 3, 1)

    base_mu = prophet_mean(1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m)
    zero_neural_mu = neural_prophet_mean(
        1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m,
        X_seasonality, X_autoregression, zeros(2), zeros(1),
    )
    @test isapprox(zero_neural_mu, base_mu)

    W_seasonality = [0.1, -0.2]
    W_autoregression = [0.5]
    neural_mu = neural_prophet_mean(
        1.0, 0.2, delta, beta, t, zeros(3), t_change, X, 0, s_a, s_m,
        X_seasonality, X_autoregression, W_seasonality, W_autoregression,
    )
    @test isapprox(
        neural_mu,
        base_mu .+ X_seasonality * W_seasonality .+ X_autoregression * W_autoregression,
    )

    nn = NeuralProphetNN(W_seasonality, W_autoregression)
    @test isapprox(
        nn(X_seasonality, X_autoregression),
        X_seasonality * W_seasonality .+ X_autoregression * W_autoregression,
    )

    nn_model = neural_prophet(
        3, 1, [0.0, 0.5, 1.0], zeros(3), [0.1, 0.2, 0.3], 1, [0.5],
        reshape([1.0, 0.0, 1.0], 3, 1), [1.0], 0.05, 0, [1.0], [0.0],
        ones(3, 2), zeros(3, 1),
    )
    @test nn_model !== nothing
end
