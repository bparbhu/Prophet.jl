using Turing
using Distributions

"""
    get_changepoint_matrix(t, t_change, T, S)

Computes the changepoint matrix.
"""
function get_changepoint_matrix(t, t_change, T, S)
    A = zeros(T, S)
    a_row = zeros(S)
    cp_idx = 1

    for i in 1:T
        while (cp_idx <= S) && (t[i] >= t_change[cp_idx])
            a_row[cp_idx] = 1
            cp_idx += 1
        end
        A[i, :] = a_row
    end
    return A
end

"""
    logistic_gamma(k, m, delta, t_change, S)

Computes logistic gamma for logistic trend functions.
"""
function logistic_gamma(k, m, delta, t_change, S)
    gamma = zeros(S)
    k_s = vcat(k, k .+ cumsum(delta))

    m_pr = m
    for i in 1:S
        gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
        m_pr += gamma[i]
    end
    return gamma
end

"""
    logistic_trend(k, m, delta, t, cap, A, t_change, S)

Computes logistic trend.
"""
function logistic_trend(k, m, delta, t, cap, A, t_change, S)
    gamma = logistic_gamma(k, m, delta, t_change, S)
    return cap .* invlogit.((k .+ A * delta) .* (t .- (m .+ A * gamma)))
end

"""
    linear_trend(k, m, delta, t, A, t_change)

Computes linear trend.
"""
function linear_trend(k, m, delta, t, A, t_change)
    return (k .+ A * delta) .* t .+ (m .+ A * (-t_change .* delta))
end

"""
    flat_trend(m, T)

Computes flat trend.
"""
function flat_trend(m, T)
    return fill(m, T)
end

@model function prophet(T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m)
    A = get_changepoint_matrix(t, t_change, T, S)
    X_sa = X .* repeat(s_a', T)
    X_sm = X .* repeat(s_m', T)

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

    y ~ MvNormal(X_sa * beta + trend .* (1 .+ X_sm * beta), sigma_obs)
end
