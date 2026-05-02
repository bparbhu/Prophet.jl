using JSON

const PROPHET_JL_SERIALIZATION_VERSION = "0.1.0"

function _serialize_value(x)
    if x === nothing || x isa Number || x isa Bool || x isa AbstractString
        return x
    elseif x isa Symbol
        return String(x)
    elseif x isa Date
        return string(x)
    elseif x isa Period
        return Dict("__period__" => string(typeof(x).name.name), "value" => Dates.value(x))
    elseif x isa AbstractVector
        return [_serialize_value(v) for v in x]
    elseif x isa AbstractMatrix
        return [[_serialize_value(x[i, j]) for j in 1:size(x, 2)] for i in 1:size(x, 1)]
    elseif x isa DataFrame
        return Dict(
            "__dataframe__" => true,
            "names" => names(x),
            "columns" => Dict(name => _serialize_value(collect(x[!, name])) for name in names(x)),
        )
    elseif x isa Dict
        return Dict(String(k) => _serialize_value(v) for (k, v) in x)
    else
        return string(x)
    end
end

function _deserialize_period(d::Dict)
    kind = d["__period__"]
    value = Int(d["value"])
    kind == "Day" && return Day(value)
    kind == "Week" && return Week(value)
    kind == "Hour" && return Hour(value)
    kind == "Minute" && return Minute(value)
    kind == "Millisecond" && return Millisecond(value)
    return Day(value)
end

function _deserialize_dataframe(d::Dict)
    cols = Pair{Symbol,Any}[]
    for name in d["names"]
        values = d["columns"][name]
        if name in ("ds", "cutoff")
            values = Date.(values)
        end
        push!(cols, Symbol(name) => values)
    end
    return DataFrame(cols...)
end

function _deserialize_value(x)
    if x isa Dict
        haskey(x, "__period__") && return _deserialize_period(x)
        haskey(x, "__dataframe__") && return _deserialize_dataframe(x)
        return Dict(String(k) => _deserialize_value(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [_deserialize_value(v) for v in x]
    else
        return x
    end
end

"""
    model_to_dict(model)

Serialize a fitted `ProphetModel` to Julia-native dictionaries and arrays. This
mirrors Python Prophet's serialize tests at the API level: fitted model state,
parameters, backend metadata, holidays, regressors, and seasonalities are
preserved without storing the backend fit object itself.
"""
function model_to_dict(m::ProphetModel)
    m.history === nothing && error("This can only be used to serialize models that have already been fit.")
    return Dict{String,Any}(
        "__prophet_jl_version" => PROPHET_JL_SERIALIZATION_VERSION,
        "growth" => m.growth,
        "changepoints" => _serialize_value(m.changepoints),
        "specified_changepoints" => m.specified_changepoints,
        "model_backend" => String(m.model_backend),
        "n_changepoints" => m.n_changepoints,
        "changepoint_range" => m.changepoint_range,
        "yearly_seasonality" => _serialize_value(m.yearly_seasonality),
        "weekly_seasonality" => _serialize_value(m.weekly_seasonality),
        "daily_seasonality" => _serialize_value(m.daily_seasonality),
        "country_holidays" => m.country_holidays,
        "holidays" => _serialize_value(m.holidays),
        "seasonality_mode" => m.seasonality_mode,
        "holidays_mode" => m.holidays_mode,
        "seasonality_prior_scale" => m.seasonality_prior_scale,
        "holidays_prior_scale" => m.holidays_prior_scale,
        "changepoint_prior_scale" => m.changepoint_prior_scale,
        "mcmc_samples" => m.mcmc_samples,
        "interval_width" => m.interval_width,
        "uncertainty_samples" => m.uncertainty_samples,
        "scaling" => m.scaling,
        "logistic_floor" => m.logistic_floor,
        "history" => _serialize_value(m.history),
        "history_dates" => _serialize_value(m.history_dates),
        "start" => _serialize_value(m.start),
        "y_min" => m.y_min,
        "y_scale" => m.y_scale,
        "t_scale" => _serialize_value(m.t_scale),
        "params" => _serialize_value(m.params),
        "fit_backend" => isnothing(m.fit_backend) ? nothing : String(m.fit_backend),
        "fit_engine" => isnothing(m.fit_engine) ? nothing : String(m.fit_engine),
        "fit_kwargs" => _serialize_value(m.fit_kwargs),
        "backend_data" => _serialize_value(m.backend_data),
        "seasonalities" => _serialize_value(m.seasonalities),
        "extra_regressors" => _serialize_value(m.extra_regressors),
        "train_component_cols" => _serialize_value(m.train_component_cols),
        "component_modes" => _serialize_value(m.component_modes),
        "train_holiday_names" => _serialize_value(m.train_holiday_names),
        "changepoints_t" => _serialize_value(m.changepoints_t),
    )
end

model_to_json(m::ProphetModel) = JSON.json(model_to_dict(m))

function model_from_dict(raw::Dict)
    d = Dict(String(k) => _deserialize_value(v) for (k, v) in raw)
    m = ProphetModel(
        growth=d["growth"],
        changepoints=get(d, "changepoints", nothing),
        model_backend=Symbol(d["model_backend"]),
        n_changepoints=Int(d["n_changepoints"]),
        changepoint_range=Float64(d["changepoint_range"]),
        yearly_seasonality=d["yearly_seasonality"],
        weekly_seasonality=d["weekly_seasonality"],
        daily_seasonality=d["daily_seasonality"],
        holidays=d["holidays"],
        country_holidays=d["country_holidays"],
        seasonality_mode=d["seasonality_mode"],
        holidays_mode=d["holidays_mode"],
        seasonality_prior_scale=Float64(d["seasonality_prior_scale"]),
        holidays_prior_scale=Float64(d["holidays_prior_scale"]),
        changepoint_prior_scale=Float64(d["changepoint_prior_scale"]),
        mcmc_samples=Int(d["mcmc_samples"]),
        interval_width=Float64(d["interval_width"]),
        uncertainty_samples=Int(d["uncertainty_samples"]),
        scaling=get(d, "scaling", "absmax"),
    )
    m.specified_changepoints = Bool(get(d, "specified_changepoints", false))
    m.logistic_floor = Bool(d["logistic_floor"])
    m.history = d["history"]
    m.history_dates = isnothing(d["history_dates"]) ? nothing : Date.(d["history_dates"])
    m.start = isnothing(d["start"]) ? nothing : Date(d["start"])
    m.y_min = isnothing(d["y_min"]) ? nothing : Float64(d["y_min"])
    m.y_scale = isnothing(d["y_scale"]) ? nothing : Float64(d["y_scale"])
    m.t_scale = d["t_scale"]
    m.params = Dict{String,Any}(String(k) => v for (k, v) in d["params"])
    m.fit_backend = isnothing(d["fit_backend"]) ? nothing : Symbol(d["fit_backend"])
    m.fit_engine = isnothing(d["fit_engine"]) ? nothing : Symbol(d["fit_engine"])
    m.fit_kwargs = Dict{String,Any}(String(k) => v for (k, v) in get(d, "fit_kwargs", Dict()))
    m.backend_data = Dict{String,Any}(String(k) => v for (k, v) in d["backend_data"])
    m.seasonalities = Dict{String,Dict{String,Any}}(
        String(k) => Dict{String,Any}(String(kk) => vv for (kk, vv) in v) for (k, v) in d["seasonalities"]
    )
    m.extra_regressors = Dict{String,Dict{String,Any}}(
        String(k) => Dict{String,Any}(String(kk) => vv for (kk, vv) in v) for (k, v) in d["extra_regressors"]
    )
    m.train_component_cols = get(d, "train_component_cols", nothing)
    raw_component_modes = get(d, "component_modes", nothing)
    m.component_modes = isnothing(raw_component_modes) ? nothing :
        Dict{String,Vector{String}}(String(k) => String.(v) for (k, v) in raw_component_modes)
    m.train_holiday_names = isnothing(d["train_holiday_names"]) ? nothing : String.(d["train_holiday_names"])
    m.changepoints_t = isnothing(d["changepoints_t"]) ? nothing : Float64.(d["changepoints_t"])
    return m
end

model_from_json(model_json::AbstractString) = model_from_dict(JSON.parse(model_json))
