using Distributions
using LinearAlgebra
using Turing

_invlogit(x) = inv(one(x) + exp(-x))

"""
    get_changepoint_matrix(t, t_change, T=length(t), S=length(t_change))

Return the same changepoint indicator matrix as Prophet's Stan model.
"""
function get_changepoint_matrix(t, t_change, T::Integer=length(t), S::Integer=length(t_change))
    A = zeros(Float64, T, S)
    cp_idx = 1
    for i in 1:T
        while cp_idx <= S && t[i] >= t_change[cp_idx]
            cp_idx += 1
        end
        if cp_idx > 1
            A[i, 1:(cp_idx - 1)] .= 1.0
        end
    end
    return A
end

function logistic_gamma(k::Real, m::Real, delta, t_change, S::Integer=length(t_change))
    gamma = zeros(Float64, S)
    k_s = vcat(k, k .+ cumsum(delta))
    m_pr = m
    for i in 1:S
        gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
        m_pr += gamma[i]
    end
    return gamma
end

function logistic_trend(k::Real, m::Real, delta, t, cap, A, t_change, S::Integer=length(t_change))
    gamma = logistic_gamma(k, m, delta, t_change, S)
    return cap .* _invlogit.((k .+ A * delta) .* (t .- (m .+ A * gamma)))
end

function linear_trend(k::Real, m::Real, delta, t, A, t_change)
    return (k .+ A * delta) .* t .+ (m .+ A * (-t_change .* delta))
end

flat_trend(m::Real, T::Integer) = fill(float(m), T)

function prophet_trend_from_A(k, m, delta, t, cap, A, t_change, trend_indicator)
    T = length(t)
    return if trend_indicator == 0
        linear_trend(k, m, delta, t, A, t_change)
    elseif trend_indicator == 1
        logistic_trend(k, m, delta, t, cap, A, t_change, length(t_change))
    elseif trend_indicator == 2
        flat_trend(m, T)
    else
        error("trend_indicator must be 0 (linear), 1 (logistic), or 2 (flat).")
    end
end

function prophet_trend(k, m, delta, t, cap, t_change, trend_indicator)
    A = get_changepoint_matrix(t, t_change, length(t), length(t_change))
    return prophet_trend_from_A(k, m, delta, t, cap, A, t_change, trend_indicator)
end

function prophet_mean_from_A(k, m, delta, beta, t, cap, A, t_change, X, trend_indicator, s_a, s_m)
    T = length(t)
    X_sa = X .* repeat(reshape(s_a, 1, :), T, 1)
    X_sm = X .* repeat(reshape(s_m, 1, :), T, 1)
    trend = prophet_trend_from_A(k, m, delta, t, cap, A, t_change, trend_indicator)

    return X_sa * beta .+ trend .* (1 .+ X_sm * beta)
end

function prophet_mean(k, m, delta, beta, t, cap, t_change, X, trend_indicator, s_a, s_m)
    T = length(t)
    S = length(t_change)
    A = get_changepoint_matrix(t, t_change, T, S)
    return prophet_mean_from_A(k, m, delta, beta, t, cap, A, t_change, X, trend_indicator, s_a, s_m)
end

"""
    prophet(T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m)

Turing implementation of the bundled Stan model. Priors and likelihood mirror
`src/stan/prophet.stan`: normal priors for `k`, `m`, and `beta`, Laplace
changepoint deltas, truncated normal observation scale, and the additive /
multiplicative regressor likelihood.
"""
@model function prophet(T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m)
    k ~ Normal(0, 5)
    m ~ Normal(0, 5)
    delta ~ filldist(Laplace(0, tau), S)
    sigma_obs ~ truncated(Normal(0, 0.5); lower=0)
    beta ~ MvNormal(zeros(K), Diagonal(sigmas .^ 2))

    mu = prophet_mean(k, m, delta, beta, t, cap, t_change, X, trend_indicator, s_a, s_m)
    y ~ MvNormal(mu, Diagonal(fill(sigma_obs^2, T)))
    return prophet_trend(k, m, delta, t, cap, t_change, trend_indicator)
end

"""
    prophet(t, X, A, t_change, s_a, s_m, tau, sigmas; cap=zeros, trend_indicator=0)

Unconditioned Turing-friendly constructor using precomputed design and
changepoint matrices. Use the `y`-accepting wrapper to condition observations.
"""
@model function prophet(
    t, X, A, t_change, s_a, s_m, tau, sigmas;
    cap=zeros(length(t)),
    trend_indicator=0,
)
    T = length(t)
    K = size(X, 2)
    S = length(t_change)

    k ~ Normal(0, 5)
    m ~ Normal(0, 5)
    delta ~ filldist(Laplace(0, tau), S)
    sigma_obs ~ truncated(Normal(0, 0.5); lower=0)
    beta ~ MvNormal(zeros(K), Diagonal(sigmas .^ 2))

    mu = prophet_mean_from_A(k, m, delta, beta, t, cap, A, t_change, X, trend_indicator, s_a, s_m)
    y ~ MvNormal(mu, Diagonal(fill(sigma_obs^2, T)))
    return prophet_trend_from_A(k, m, delta, t, cap, A, t_change, trend_indicator)
end

function prophet(
    t, X, y, A, t_change, s_a, s_m, tau, sigmas;
    cap=zeros(length(t)),
    trend_indicator=0,
)
    return condition(
        prophet(t, X, A, t_change, s_a, s_m, tau, sigmas; cap=cap, trend_indicator=trend_indicator),
        y=y,
    )
end
