import CairoMakie
using DataFrames
using Dates
import Gadfly

function _history_frame(m)
    hasproperty(m, :history) || error("plot_forecast expects a model with a `history` field.")
    return getproperty(m, :history)
end

function _uncertainty_samples(m)
    hasproperty(m, :uncertainty_samples) ? getproperty(m, :uncertainty_samples) : 0
end

function _logistic_floor(m)
    hasproperty(m, :logistic_floor) ? getproperty(m, :logistic_floor) : false
end

function _makie_x(ds)
    return Dates.value.(Date.(ds) .- Date(1970, 1, 1))
end

function _makie_axis!(ax, ds)
    ticks = _makie_x(ds)
    isempty(ticks) && return ax
    step = max(1, cld(length(ticks), 6))
    selected = ticks[1:step:end]
    labels = Dates.format.(Date(1970, 1, 1) .+ Day.(selected), dateformat"yyyy-mm-dd")
    ax.xticks = (selected, labels)
    return ax
end

function _plot_forecast_makie(
    m, fcst; uncertainty=true, plot_cap=true, xlabel="ds", ylabel="y", include_legend=false
)
    history = _history_frame(m)
    fig = CairoMakie.Figure(size=(900, 520))
    ax = CairoMakie.Axis(fig[1, 1], xlabel=xlabel, ylabel=ylabel)
    hx = _makie_x(history.ds)
    fx = _makie_x(fcst.ds)

    CairoMakie.scatter!(ax, hx, history.y; color=:black, markersize=7, label="Observed")
    CairoMakie.lines!(ax, fx, fcst.yhat; color=:steelblue, linewidth=2, label="Forecast")

    if uncertainty && _uncertainty_samples(m) > 0 &&
            all(x -> x in names(fcst), ["yhat_lower", "yhat_upper"])
        CairoMakie.band!(
            ax, fx, fcst.yhat_lower, fcst.yhat_upper;
            color=(:steelblue, 0.18), label="Interval"
        )
    end

    if plot_cap && "cap" in names(fcst)
        CairoMakie.lines!(ax, fx, fcst.cap; color=:black, linestyle=:dash, label="Cap")
    end
    if plot_cap && _logistic_floor(m) && "floor" in names(fcst)
        CairoMakie.lines!(ax, fx, fcst.floor; color=:black, linestyle=:dash, label="Floor")
    end
    _makie_axis!(ax, vcat(Date.(history.ds), Date.(fcst.ds)))
    include_legend && CairoMakie.axislegend(ax)
    return fig
end

function _plot_forecast_gadfly(
    m, fcst; uncertainty=true, plot_cap=true, xlabel="ds", ylabel="y", include_legend=false
)
    history = _history_frame(m)
    layers = Any[
        Gadfly.layer(history, x=:ds, y=:y, Gadfly.Geom.point, Gadfly.Theme(default_color="black")),
        Gadfly.layer(fcst, x=:ds, y=:yhat, Gadfly.Geom.line, Gadfly.Theme(default_color="steelblue")),
    ]

    if uncertainty && _uncertainty_samples(m) > 0 &&
            all(x -> x in names(fcst), ["yhat_lower", "yhat_upper"])
        push!(
            layers,
            Gadfly.layer(
                fcst, x=:ds, ymin=:yhat_lower, ymax=:yhat_upper,
                Gadfly.Geom.ribbon, Gadfly.Theme(default_color="steelblue", alphas=[0.18])
            ),
        )
    end

    if plot_cap && "cap" in names(fcst)
        push!(layers, Gadfly.layer(fcst, x=:ds, y=:cap, Gadfly.Geom.line(style=:dash)))
    end
    if plot_cap && _logistic_floor(m) && "floor" in names(fcst)
        push!(layers, Gadfly.layer(fcst, x=:ds, y=:floor, Gadfly.Geom.line(style=:dash)))
    end

    guides = Any[Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel)]
    if include_legend
        push!(guides, Gadfly.Guide.manual_color_key("Legend", ["Observed", "Forecast"], ["black", "steelblue"]))
    end
    return Gadfly.plot(layers..., guides...)
end

"""
    plot_forecast(m, fcst; backend=:makie, ...)

Plot observed history and forecast values. Supported backends are `:makie` and
`:gadfly`.
"""
function plot_forecast(m, fcst; backend::Symbol=:makie, kwargs...)
    if backend == :makie
        return _plot_forecast_makie(m, fcst; kwargs...)
    elseif backend == :gadfly
        return _plot_forecast_gadfly(m, fcst; kwargs...)
    else
        error("Unsupported plotting backend $backend. Use :makie or :gadfly.")
    end
end

function _plot_forecast_component_makie(m, fcst, name; uncertainty=true, plot_cap=false)
    col = Symbol(name)
    fig = CairoMakie.Figure(size=(900, 420))
    ax = CairoMakie.Axis(fig[1, 1], xlabel="ds", ylabel=String(name))
    fx = _makie_x(fcst.ds)
    CairoMakie.lines!(ax, fx, fcst[!, col]; color=:steelblue, linewidth=2)

    lower = Symbol(string(name, "_lower"))
    upper = Symbol(string(name, "_upper"))
    if uncertainty && _uncertainty_samples(m) > 0 && all(x -> String(x) in names(fcst), [lower, upper])
        CairoMakie.band!(ax, fx, fcst[!, lower], fcst[!, upper]; color=(:steelblue, 0.18))
    end
    if plot_cap && "cap" in names(fcst)
        CairoMakie.lines!(ax, fx, fcst.cap; color=:black, linestyle=:dash)
    end
    _makie_axis!(ax, fcst.ds)
    return fig
end

function _plot_forecast_component_gadfly(m, fcst, name; uncertainty=true, plot_cap=false)
    col = Symbol(name)
    layers = Any[
        Gadfly.layer(fcst, x=:ds, y=col, Gadfly.Geom.line, Gadfly.Theme(default_color="steelblue")),
    ]
    lower = Symbol(string(name, "_lower"))
    upper = Symbol(string(name, "_upper"))
    if uncertainty && _uncertainty_samples(m) > 0 && all(x -> String(x) in names(fcst), [lower, upper])
        push!(
            layers,
            Gadfly.layer(
                fcst, x=:ds, ymin=lower, ymax=upper,
                Gadfly.Geom.ribbon, Gadfly.Theme(default_color="steelblue", alphas=[0.18])
            ),
        )
    end
    if plot_cap && "cap" in names(fcst)
        push!(layers, Gadfly.layer(fcst, x=:ds, y=:cap, Gadfly.Geom.line(style=:dash)))
    end
    return Gadfly.plot(layers..., Gadfly.Guide.xlabel("ds"), Gadfly.Guide.ylabel(String(name)))
end

function plot_forecast_component(m, fcst, name; backend::Symbol=:makie, kwargs...)
    if backend == :makie
        return _plot_forecast_component_makie(m, fcst, name; kwargs...)
    elseif backend == :gadfly
        return _plot_forecast_component_gadfly(m, fcst, name; kwargs...)
    else
        error("Unsupported plotting backend $backend. Use :makie or :gadfly.")
    end
end
