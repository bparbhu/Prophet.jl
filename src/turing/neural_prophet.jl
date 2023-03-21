using Turing
using Distributions
using Flux

# Define the Neural Prophet neural network structure
struct NeuralProphetNN{T <: AbstractArray}
    W_seasonality::T
    W_autoregression::T
end

function (nn::NeuralProphetNN)(X_seasonality, X_autoregression)
    y_seasonality = nn.W_seasonality * X_seasonality
    y_autoregression = nn.W_autoregression * X_autoregression
    return y_seasonality .+ y_autoregression
end

# Create a custom loss function
function custom_loss(y_pred, y_true, W_seasonality, W_autoregression, sigma)
    nll = -logpdf(MvNormal(y_pred, sigma), y_true)
    return nll
end

# Perform Bayesian inference using the Turing.jl model with Flux NN
@model function neural_prophet(T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m, X_seasonality, X_autoregression, H, sigma)
    A = get_changepoint_matrix(t, t_change, T, S)
    X_sa = X .* repeat(s_a', T)
    X_sm = X .* repeat(s_m', T)

    # Priors for the weights and biases
    W_seasonality ~ MvNormal(zeros(K), I)
    W_autoregression ~ MvNormal(zeros(H), I)

    k ~ Normal(0, 5)
    m ~ Normal(0, 5)
    delta ~ Laplace(0, tau)
    sigma_obs ~ Truncated(Normal(0, 0.5), 0, Inf)
    beta ~ MvNormal(fill(0.0, K), diagm(0 => sigmas))

    if trend_indicator == 0
        trend = linear_trend(k, m, delta, t, A, t_change)
    elseif trend_indicator == 1
        trend = logistic_trend(k, m, delta, t, cap, A, t_change, S)
    elseif trend_indicator == 2
        trend = flat_trend(m, T)
    end

    # Create a NeuralProphetNN instance
    nn = NeuralProphetNN(W_seasonality, W_autoregression)

    # Compute the predicted outputs
    Y_pred = nn.(X_seasonality, X_autoregression)

    y ~ MvNormal(X_sa * beta + trend .* (1 .+ X_sm * beta) .+ Y_pred, sigma_obs)
end
