@enum TrendIndicator LINEAR=0 LOGISTIC=1 FLAT=2

struct ModelInputData
    T::Int
    S::Int
    K::Int
    tau::Float64
    trend_indicator::Int
    y::Vector{Float64}
    t::Vector{Float64}
    cap::Vector{Float64}
    t_change::Vector{Float64}
    s_a::Vector{Float64}
    s_m::Vector{Float64}
    X::Matrix{Float64}
    sigmas::Vector{Float64}
end

struct ModelParams
    k::Float64
    m::Float64
    delta::Vector{Float64}
    beta::Vector{Float64}
    sigma_obs::Float64
end

trend_indicator(growth::AbstractString) = Dict(
    "linear" => Int(LINEAR),
    "logistic" => Int(LOGISTIC),
    "flat" => Int(FLAT),
)[String(growth)]

function model_input_data(data::Dict{String,Any})
    return ModelInputData(
        Int(data["T"]),
        Int(data["S"]),
        Int(data["K"]),
        Float64(data["tau"]),
        Int(data["trend_indicator"]),
        Float64.(data["y"]),
        Float64.(data["t"]),
        Float64.(data["cap"]),
        Float64.(data["t_change"]),
        Float64.(data["s_a"]),
        Float64.(data["s_m"]),
        Matrix{Float64}(data["X"]),
        Float64.(data["sigmas"]),
    )
end

function model_input_data_dict(data::ModelInputData)
    return Dict{String,Any}(
        "T" => data.T,
        "S" => data.S,
        "K" => data.K,
        "tau" => data.tau,
        "trend_indicator" => data.trend_indicator,
        "y" => data.y,
        "t" => data.t,
        "cap" => data.cap,
        "t_change" => data.t_change,
        "s_a" => data.s_a,
        "s_m" => data.s_m,
        "X" => data.X,
        "sigmas" => data.sigmas,
    )
end

function model_params_dict(params::ModelParams)
    return Dict{String,Any}(
        "k" => params.k,
        "m" => params.m,
        "delta" => params.delta,
        "beta" => params.beta,
        "sigma_obs" => params.sigma_obs,
    )
end

function sanitize_custom_inits(default_inits::Dict{String,Any}, custom_inits::Dict{String,Any})
    sanitized = Dict{String,Any}()
    for param in ("k", "m", "sigma_obs")
        sanitized[param] = try
            Float64(custom_inits[param])
        catch
            default_inits[param]
        end
    end
    for param in ("delta", "beta")
        default_value = Float64.(default_inits[param])
        custom_value = get(custom_inits, param, default_value)
        sanitized[param] = size(custom_value) == size(default_value) ? Float64.(custom_value) : default_value
    end
    return sanitized
end

function stan_to_dict(column_names, values)
    names = String.(column_names)
    data = values isa AbstractMatrix ? values : reshape(values, 1, :)
    output = Dict{String,Any}()
    current = split(replace(names[1], "[" => "."), ".")[1]
    start = 1
    for (idx, name) in enumerate(names)
        parsed = split(replace(name, "[" => "."), ".")[1]
        if parsed != current
            haskey(output, current) && error("Found repeated column name.")
            block = data[:, start:(idx - 1)]
            output[current] = size(block, 2) == 1 ? vec(block) : block
            current = parsed
            start = idx
        end
    end
    haskey(output, current) && error("Found repeated column name.")
    block = data[:, start:end]
    output[current] = size(block, 2) == 1 ? vec(block) : block
    return output
end
