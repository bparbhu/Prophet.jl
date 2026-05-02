using Distributions
using Flux
using LinearAlgebra
using Turing

struct NeuralProphetNN
    W_seasonality
    W_autoregression
end

function (nn::NeuralProphetNN)(X_seasonality, X_autoregression)
    return X_seasonality * nn.W_seasonality .+ X_autoregression * nn.W_autoregression
end

function neural_prophet_mean(
    k, m, delta, beta, t, cap, t_change, X, trend_indicator, s_a, s_m,
    X_seasonality, X_autoregression, W_seasonality, W_autoregression,
)
    base_mu = prophet_mean(k, m, delta, beta, t, cap, t_change, X, trend_indicator, s_a, s_m)
    nn = NeuralProphetNN(W_seasonality, W_autoregression)
    return base_mu .+ nn(X_seasonality, X_autoregression)
end

"""
    neural_prophet(...)

Flux/Turing extension of the Stan-equivalent Prophet model. The trend,
changepoint, regressor, and observation priors are the same as `prophet`; the
additional Bayesian neural-network term is an additive residual component.
"""
@model function neural_prophet(
    T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m,
    X_seasonality, X_autoregression,
)
    n_seasonality = size(X_seasonality, 2)
    n_autoregression = size(X_autoregression, 2)

    W_seasonality ~ MvNormal(zeros(n_seasonality), Diagonal(ones(n_seasonality)))
    W_autoregression ~ MvNormal(zeros(n_autoregression), Diagonal(ones(n_autoregression)))

    k ~ Normal(0, 5)
    m ~ Normal(0, 5)
    delta ~ filldist(Laplace(0, tau), S)
    sigma_obs ~ truncated(Normal(0, 0.5); lower=0)
    beta ~ MvNormal(zeros(K), Diagonal(sigmas .^ 2))

    mu = neural_prophet_mean(
        k, m, delta, beta, t, cap, t_change, X, trend_indicator, s_a, s_m,
        X_seasonality, X_autoregression, W_seasonality, W_autoregression,
    )
    y ~ MvNormal(mu, Diagonal(fill(sigma_obs^2, T)))
end
