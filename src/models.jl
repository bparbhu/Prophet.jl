using AbstractTrees
using Stan
using DataFrames
using OrderedCollections
using Logging
using Platform

const logger = Logging.current_logger()

const PLATFORM = Sys.iswindows() ? "win" : "unix"

abstract type IStanBackend end

function set_options!(backend::IStanBackend; kwargs...)
    for (k, v) in kwargs
        if k == :newton_fallback
            backend.newton_fallback = v
        else
            throw(ArgumentError("Unknown option $k"))
        end
    end
end

mutable struct StanBackend <: IStanBackend
    model
    stan_fit
    newton_fallback::Bool
end

StanBackend() = StanBackend(nothing, nothing, true)


function load_model()
    model_file = p"$(dirname(@__FILE__))/stan_model/prophet_model.stan"
    return Stanmodel(name="prophet_model", model=model_file)
end


function fit(model, stan_init, stan_data; kwargs...)
    if !haskey(kwargs, :inits) && haskey(kwargs, :init)
        stan_init = sanitize_custom_inits(stan_init, kwargs[:init])
        delete!(kwargs, :init)
    end

    inits_list, data_list = prepare_data(stan_init, stan_data)
    args = Dict(
        :data => data_list,
        :init => inits_list,
        :algorithm => data_list["T"] < 100 ? "Newton" : "LBFGS",
        :iter => 10000,
    )
    merge!(args, kwargs)

    try
        stan_fit = optimize(model, args)
    catch e
        if !args[:algorithm] == "Newton"
            throw(e)
        end
        Logging.warn(logger, "Optimization terminated abnormally. Falling back to Newton.")
        args[:algorithm] = "Newton"
        stan_fit = optimize(model, args)
    end

    params = stan_to_dict_array(stan_fit.names, stan_fit.optimized_params)
    for par in keys(params)
        params[par] = reshape(params[par], (1, :))
    end

    return params
end

function sampling(model, stan_init, stan_data, samples; kwargs...)
    if !haskey(kwargs, :inits) && haskey(kwargs, :init)
        stan_init = sanitize_custom_inits(stan_init, kwargs[:init])
        delete!(kwargs, :init)
    end

    inits_list, data_list = prepare_data(stan_init, stan_data)
    args = Dict(
        :data => data_list,
        :init => inits_list,
    )
    if !haskey(kwargs, :chains)
        kwargs[:chains] = 4
    end
    iter_half = div(samples, 2)
    kwargs[:iter_sampling] = iter_half
    if !haskey(kwargs, :iter_warmup)
        kwargs[:iter_warmup] = iter_half
    end
    merge!(args, kwargs)

    stan_fit = sample(model, args)
    res = stan_fit.draws
    (samples, c, columns) = size(res)
    res = reshape(res, (samples * c, columns))
    params = stan_to_dict_numpy(stan_fit.names, res)

    for par in keys(params)
        s = size(params[par])
        if s[2] == 1
            params[par] = reshape(params[par], (s[1],))
        end

        if par in ["delta", "beta"] && length(s) < 2
            params[par] = reshape(params[par], (-1, 1))
        end
    end

    return params
end


function sanitize_custom_inits(default_inits, custom_inits)
    sanitized = Dict{String, Any}()
    for param in ["k", "m", "sigma_obs"]
        try
            sanitized[param] = Float64(custom_inits[param])
        catch
            sanitized[param] = default_inits[param]
        end
    end

    for param in ["delta", "beta"]
        if size(default_inits[param]) == size(custom_inits[param])
            sanitized[param] = custom_inits[param]
        else
            sanitized[param] = default_inits[param]
        end
    end

    return sanitized
end

function prepare_data(init, data)
    cmdstan_data = Dict(
        "T" => data["T"],
        "S" => data["S"],
        "K" => data["K"],
        "tau" => data["tau"],
        "trend_indicator" => data["trend_indicator"],
        "y" => data["y"] |> vec |> collect,
        "t" => data["t"] |> vec |> collect,
        "cap" => data["cap"] |> vec |> collect,
        "t_change" => data["t_change"] |> vec |> collect,
        "s_a" => data["s_a"] |> vec |> collect,
        "s_m" => data["s_m"] |> vec |> collect,
        "X" => data["X"] |> Matrix |> collect,
        "sigmas" => data["sigmas"],
    )

    cmdstan_init = Dict(
        "k" => init["k"],
        "m" => init["m"],
        "delta" => init["delta"] |> vec |> collect,
        "beta" => init["beta"] |> vec |> collect,
        "sigma_obs" => init["sigma_obs"],
    )

    return (cmdstan_init, cmdstan_data)
end

function stan_to_dict_array(column_names, data)
    output = OrderedDict{String, Array}()

    prev = nothing
    start = 1
    end_ = 1
    two_dims = ndims(data) > 1

    for cname in column_names
        parsed = occursin(".", cname) ? split(cname, ".") : split(cname, "[")
        curr = parsed[1]

        if prev === nothing
            prev = curr
        end

        if curr != prev
            if haskey(output, prev)
                throw("Found repeated column name")
            end

            if two_dims
                output[prev] = data[:, start:end_ - 1]
            else
                output[prev] = data[start:end_ - 1]
            end

            prev = curr
            start = end_
        end

        end_ += 1
    end

    if haskey(output, prev)
        throw("Found repeated column name")
    end

    if two_dims
        output[prev] = data[:, start:end_ - 1]
    else
        output[prev] = data[start:end_ - 1]
    end

    return output
end

struct StanBackendEnum
    value::Type{<:IStanBackend}
end

get_backend_class(name::String)::IStanBackend = StanBackendEnum(Symbol(name)).value

StanBackendEnum(:STAN) = StanBackendEnum(StanBackend)