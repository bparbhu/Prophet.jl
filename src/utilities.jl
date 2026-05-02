using DataFrames
using Statistics

function _param_vector(m::ProphetModel, name::String)
    value = m.params[name]
    value isa Number && return [Float64(value)]
    return Float64.(vec(value))
end

function warm_start_params(m::ProphetModel)
    m.history === nothing && error("Model has not been fit.")
    return Dict{String,Any}(
        "k" => _param_vector(m, "k")[1],
        "m" => _param_vector(m, "m")[1],
        "sigma_obs" => _param_vector(m, "sigma_obs")[1],
        "delta" => _param_vector(m, "delta"),
        "beta" => _param_vector(m, "beta"),
    )
end

function regressor_index(m::ProphetModel, name::AbstractString)
    m.history === nothing && error("Model has not been fit.")
    feature_names = String.(get(m.backend_data, "feature_names", String[]))
    idx = findfirst(==(String(name)), feature_names)
    idx === nothing && error("Regressor \"$name\" not found.")
    return idx
end

function regressor_coefficients(m::ProphetModel)
    m.history === nothing && error("Model has not been fit.")
    isempty(m.extra_regressors) && error("No extra regressors found.")
    beta = _param_vector(m, "beta")
    rows = NamedTuple[]
    for (name, props) in sort(collect(m.extra_regressors); by=first)
        idx = regressor_index(m, name)
        raw_coef = beta[idx]
        coef = props["mode"] == "additive" ?
            raw_coef * m.y_scale / Float64(props["std"]) :
            raw_coef / Float64(props["std"])
        push!(
            rows,
            (
                regressor=name,
                regressor_mode=String(props["mode"]),
                center=Float64(props["mu"]),
                coef_lower=coef,
                coef=coef,
                coef_upper=coef,
            ),
        )
    end
    return DataFrame(rows)
end
