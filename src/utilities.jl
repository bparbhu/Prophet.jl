using DataFrames
using Statistics
using JSON

function regressor_index(m, name)
    return findfirst(m.train_component_cols[name] .== 1)
end

function regressor_coefficients(m)
    @assert length(m.extra_regressors) > 0 "No extra regressors found."
    coefs = []
    for (regressor, params) in m.extra_regressors
        beta = m.params["beta"][:, regressor_index(m, regressor)]
        if params["mode"] == "additive"
            coef = beta * m.y_scale / params["std"]
        else
            coef = beta / params["std"]
        end
        percentiles = [
            (1 - m.interval_width) / 2,
            1 - (1 - m.interval_width) / 2,
        ]
        coef_bounds = quantile(coef, percentiles)
        record = Dict(
            "regressor" => regressor,
            "regressor_mode" => params["mode"],
            "center" => params["mu"],
            "coef_lower" => coef_bounds[1],
            "coef" => mean(coef),
            "coef_upper" => coef_bounds[2],
        )
        push!(coefs, record)
    end
    return DataFrame(coefs)
end

function warm_start_params(m)
    res = Dict()
    for pname in ["k", "m", "sigma_obs"]
        if m.mcmc_samples == 0
            res[pname] = m.params[pname][1]
        else
            res[pname] = mean(m.params[pname])
        end
    end
    for pname in ["delta", "beta"]
        if m.mcmc_samples == 0
            res[pname] = m.params[pname][:]
        else
            res[pname] = mean(m.params[pname], dims=1)
        end
    end
    return res
end
