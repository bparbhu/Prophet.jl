using Gadfly
using DataFrames
using Dates

function plot_forecast(
    m, fcst; uncertainty=true, plot_cap=true, xlabel="ds", ylabel="y", include_legend=false
)
    # Create a new DataFrame for the plot
    plot_data = DataFrame(
        ds = vcat(m.history[:, :ds], fcst[:, :ds]),
        y = vcat(m.history[:, :y], fill(NaN, size(fcst, 1))),
        yhat = vcat(fill(NaN, size(m.history, 1)), fcst[:, :yhat]),
    )

    # Define layers
    layers = [
        layer(plot_data, x=:ds, y=:y, Geom.point, Theme(default_color=colorant"black")),
        layer(plot_data, x=:ds, y=:yhat, Geom.line, Theme(default_color=colorant"#0072B2"))
    ]

    # Add uncertainty interval
    if uncertainty && m.uncertainty_samples > 0
        push!(layers,
            layer(
                fcst, x=:ds, ymin=:yhat_lower, ymax=:yhat_upper,
                Geom.ribbon, Theme(default_color=colorant"#0072B2", alpha=0.2)
            )
        )
    end

    # Add capacity lines
    if plot_cap && "cap" in names(fcst)
        push!(layers,
            layer(
                fcst, x=:ds, y=:cap, Geom.line(style=:dash),
                Theme(default_color=colorant"black")
            )
        )
    end

    if m.logistic_floor && "floor" in names(fcst) && plot_cap
        push!(layers,
            layer(
                fcst, x=:ds, y=:floor, Geom.line(style=:dash),
                Theme(default_color=colorant"black")
            )
        )
    end

    # Create the plot
    p = plot(
        layers...,
        Scale.x_datetime(format=Dates.dateformat"yyyy-mm-dd"),
        Guide.xlabel(xlabel),
        Guide.ylabel(ylabel),
        Guide.title("Forecast Plot")
    )

    # Add legend
    if include_legend
        p = plot(p, Guide.manual_color_key("Legend",
            ["Observed data points", "Forecast", "Uncertainty interval"],
            ["black", "#0072B2", "#0072B2"]
        ))
    end

    return p
end


function plot_forecast_component(
    m, fcst, name; uncertainty=true, plot_cap=false
)
    # Create a new DataFrame for the plot
    plot_data = DataFrame(
        ds = fcst[:, :ds],
        component = fcst[:, Symbol(name)]
    )

    # Define layers
    layers = [
        layer(plot_data, x=:ds, y=:component, Geom.line, Theme(default_color=colorant"#0072B2"))
    ]

    # Add capacity lines
    if plot_cap && "cap" in names(fcst)
        push!(layers,
            layer(
                fcst, x=:ds, y=:cap, Geom.line(style=:dash),
                Theme(default_color=colorant"black")
            )
        )
    end

    if m.logistic_floor && "floor" in names(fcst) && plot_cap
        push!(layers,
            layer(
                fcst, x=:ds, y=:floor, Geom.line(style=:dash),
                Theme(default_color=colorant"black")
            )
        )
    end

    # Add uncertainty interval
    if uncertainty && m.uncertainty_samples > 0
        push!(layers,
            layer(
                fcst, x=:ds, ymin=Symbol("$(name)_lower"), ymax=Symbol("$(name)_upper"),
                Geom.ribbon, Theme(default_color=colorant"#0072B2", alpha=0.2)
            )
        )
    end

    # Create the plot
    p = plot(
        layers...,
        Scale.x_datetime(format=Dates.dateformat"yyyy-mm-dd"),
        Guide.xlabel("ds"),
        Guide.ylabel(name),
        Guide.title("Forecast Component: $name")
    )

    return p
end
