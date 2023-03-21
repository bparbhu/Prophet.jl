using JSON
using OrderedCollections
using DataFrames
using Dates
using TimeZones

include("path/to/prophet.stan")  # Replace this with the path to your Prophet.jl file
using .Prophet

const PROPHET_VERSION = "0.1"  # Update this according to your Julia implementation's version

const SIMPLE_ATTRIBUTES = [
    :growth, :n_changepoints, :specified_changepoints, :changepoint_range,
    :yearly_seasonality, :weekly_seasonality, :daily_seasonality,
    :seasonality_mode, :seasonality_prior_scale, :changepoint_prior_scale,
    :holidays_prior_scale, :mcmc_samples, :interval_width, :uncertainty_samples,
    :y_scale, :logistic_floor, :country_holidays, :component_modes
]

const PD_SERIES = [:changepoints, :history_dates, :train_holiday_names]

const PD_TIMESTAMP = [:start]

const PD_TIMEDELTA = [:t_scale]

const PD_DATAFRAME = [:holidays, :history, :train_component_cols]

const NP_ARRAY = [:changepoints_t]

const ORDEREDDICT = [:seasonalities, :extra_regressors]

# The rest of the code adapted to work with the local Julia implementation of Prophet

function model_to_dict(model)
    if isnothing(model.history)
        error("This can only be used to serialize models that have already been fit.")
    end

    model_dict = Dict(attribute => getfield(model, attribute) for attribute in SIMPLE_ATTRIBUTES)
    for attribute in PD_SERIES
        if isnothing(getfield(model, attribute))
            model_dict[attribute] = nothing
        else
            model_dict[attribute] = JSON.json(getfield(model, attribute), dateformat="iso")
        end
    end

    for attribute in PD_TIMESTAMP
        model_dict[attribute] = Dates.datetime2unix(getfield(model, attribute))
    end

    for attribute in PD_TIMEDELTA
        model_dict[attribute] = getfield(model, attribute).value / 1000
    end

    for attribute in PD_DATAFRAME
        if isnothing(getfield(model, attribute))
            model_dict[attribute] = nothing
        else
            model_dict[attribute] = JSON.json(getfield(model, attribute), orient="table")
        end
    end

    for attribute in NP_ARRAY
        model_dict[attribute] = vec(getfield(model, attribute))
    end

    for attribute in ORDEREDDICT
        model_dict[attribute] = [collect(keys(getfield(model, attribute))), getfield(model, attribute)]
    end

    fit_kwargs = deepcopy(model.fit_kwargs)
    if haskey(fit_kwargs, "init")
        for (k, v) in fit_kwargs["init"]
            if isa(v, Vector)
                fit_kwargs["init"][k] = vec(v)
            elseif isa(v, AbstractFloat)
                fit_kwargs["init"][k] = Float64(v)
            end
        end
    end
    model_dict["fit_kwargs"] = fit_kwargs
    model_dict["params"] = Dict(k => vec(v) for (k, v) in model.params)
    model_dict["__prophet_version"] = PROPHET_VERSION
    return model_dict
end

model_to_json(model) = JSON.json(model_to_dict(model))

function model_from_dict(model_dict)
    model = Prophet.Prophet()  # Make sure to use the correct constructor for your Julia implementation
    for attribute in SIMPLE_ATTRIBUTES
        setfield!(model, attribute, model_dict[attribute])
    end
    for attribute in PD_SERIES
        if isnothing(model_dict[attribute])
            setfield!(model, attribute, nothing)
        else
            s = JSON.parse(String, model_dict[attribute], typ="series", orient="split")
            if s.name == "ds"
                if isempty(s)
                    s = Dates.DateTime.(s)
                end
                s = TimeZones.localzone()(s)
            end
            setfield!(model, attribute, s)
        end
    end

    for attribute in PD_TIMESTAMP
        setfield!(model, attribute, Dates.unix2datetime(model_dict[attribute]))
    end

    for attribute in PD_TIMEDELTA
        setfield!(model, attribute, Dates.Millisecond(model_dict[attribute] * 1000))
    end

    for attribute in PD_DATAFRAME
        if isnothing(model_dict[attribute])
            setfield!(model, attribute, nothing)
        else
            df = JSON.parse(String, model_dict[attribute], typ="frame", orient="table", convert_dates=["ds"])
            if attribute == :train_component_cols
                # Special handling because of named index column
                df = rename!(df, Dict(:component => "component", :col => "col"))
            end
            setfield!(model, attribute, df)
        end
    end

    for attribute in NP_ARRAY
        setfield!(model, attribute, Array{Float64}(model_dict[attribute]))
    end

    for attribute in ORDEREDDICT
        key_list, unordered_dict = model_dict[attribute]
        od = OrderedDict()
        for key in key_list
            od[key] = unordered_dict[key]
        end
        setfield!(model, attribute, od)
    end

    model.fit_kwargs = model_dict["fit_kwargs"]
    model.params = Dict(k => Array{Float64}(v) for (k, v) in model_dict["params"])
    model.stan_backend = nothing
    model.stan_fit = nothing
    return model
end

model_from_json(model_json) = model_from_dict(JSON.parse(String, model_json))

