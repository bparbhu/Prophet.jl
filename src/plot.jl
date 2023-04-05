using Gadfly
using DataFrames
using Dates
using Plotly
using Printf
using Statistics

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


function plot_forecast_component(m, fcst, name;
                                 uncertainty=true, plot_cap=false, figsize=(10, 6))
    fcst_t = convert(Array, fcst[:ds])

    trace = scatter(
        x=fcst_t,
        y=fcst[name],
        mode="lines",
        line=Dict("color" => "#0072B2"),
        name=name
    )

    data = [trace]

    if plot_cap && "cap" in names(fcst)
        cap_trace = scatter(
            x=fcst_t,
            y=fcst[:cap],
            mode="lines",
            line=Dict("color" => "black", "dash" => "dash"),
            name="cap"
        )
        push!(data, cap_trace)
    end

    if m.logistic_floor && "floor" in names(fcst) && plot_cap
        floor_trace = scatter(
            x=fcst_t,
            y=fcst[:floor],
            mode="lines",
            line=Dict("color" => "black", "dash" => "dash"),
            name="floor"
        )
        push!(data, floor_trace)
    end

    if uncertainty && m.uncertainty_samples > 0
        uncertainty_trace = scatter(
            x=fcst_t,
            y=fcst[Symbol(name * "_upper")],
            mode="lines",
            line=Dict("color" => "transparent"),
            fill="tonexty",
            fillcolor="rgba(0, 114, 178, 0.2)",
            showlegend=false
        )
        push!(data, uncertainty_trace)

        uncertainty_trace_lower = scatter(
            x=fcst_t,
            y=fcst[Symbol(name * "_lower")],
            mode="lines",
            line=Dict("color" => "transparent"),
            showlegend=false
        )
        push!(data, uncertainty_trace_lower)
    end

    layout = Layout(
        xaxis=Dict(
            "title" => "ds",
            "showgrid" => true,
            "gridcolor" => "gray",
            "gridwidth" => 1,
            "gridalpha" => 0.2
        ),
        yaxis=Dict(
            "title" => name,
            "showgrid" => true,
            "gridcolor" => "gray",
            "gridwidth" => 1,
            "gridalpha" => 0.2
        ),
        height=figsize[2] * 60,
        width=figsize[1] * 60
    )

    if name in m.component_modes["multiplicative"]
        layout["yaxis"] = merge(layout["yaxis"], Dict("tickformat" => ",.0%"))
    end

    return plot(data, layout)
end


function seasonality_plot_df(m, ds)
    df_dict = Dict("ds" => ds, "cap" => 1.0, "floor" => 0.0)

    for name in keys(m.extra_regressors)
        df_dict[name] = 0.0
    end

    # Activate all conditional seasonality columns
    for props in values(m.seasonalities)
        if !isnothing(props["condition_name"])
            df_dict[props["condition_name"]] = true
        end
    end

    df = DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df
end


function plot_weekly_gadfly(m; uncertainty=true, weekly_start=0, name="weekly")
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = Date(2017, 1, 1) .+ Day.(0:6) .+ Day(weekly_start)
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days_of_week = Dates.format.(days, "E")

    plot_data = DataFrame(Day_of_week = days_of_week, Seasonality = seas[name])

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]
        plot_data[!, Symbol(name * "_lower")] = lower
        plot_data[!, Symbol(name * "_upper")] = upper
        plot_layer = layer(
            x=:Day_of_week, y=:Seasonality, ymin=:lower, ymax=:upper,
            Geom.line, Geom.ribbon, Theme(default_color=colorant"#0072B2")
        )
    else
        plot_layer = layer(plot_data, x=:Day_of_week, y=:Seasonality, Geom.line,
                           Theme(default_color=colorant"#0072B2"))
    end

    p = plot(plot_layer,
             Guide.xlabel("Day of week"),
             Guide.ylabel(name),
             Coord.cartesian(xmin=1, xmax=7),
             Theme(grid_line_width=1, grid_line_color=colorant"gray", alpha=0.2)
    )

    return p
end


function plot_weekly_plotly(m; uncertainty=true, weekly_start=0, name="weekly")
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = Date(2017, 1, 1) .+ Day.(0:6) .+ Day(weekly_start)
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days_of_week = Dates.format.(days, "E")

    plot_data = DataFrame(Day_of_week = days_of_week, Seasonality = seas[name])

    trace = scatter(x=plot_data.Day_of_week, y=plot_data.Seasonality, mode="lines", 
                    line=Dict(:color => "rgb(0, 114, 178)"))

    layout = Layout(
        xaxis=Dict(:title => "Day of week"),
        yaxis=Dict(:title => name),
        showlegend=false,
        margin=Dict(:l => 50, :r => 50, :b => 50, :t => 50),
        plot_bgcolor="white"
    )

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]
        plot_data[!, Symbol(name * "_lower")] = lower
        plot_data[!, Symbol(name * "_upper")] = upper
        
        fill_trace = scatter(x=plot_data.Day_of_week, y=lower, mode="lines",
                             line=Dict(:color => "rgba(0, 114, 178, 0)"),
                             showlegend=false)
        fill_trace2 = scatter(x=plot_data.Day_of_week, y=upper, mode="lines",
                              fill="tonexty",
                              line=Dict(:color => "rgba(0, 114, 178, 0.2)"),
                              showlegend=false)

        plot_data = [trace, fill_trace, fill_trace2]
    else
        plot_data = [trace]
    end

    p = PlotlyJS.plot(plot_data, layout)

    return p
end


function plot_yearly_gadfly(m; uncertainty=true, yearly_start=0, name="yearly")
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = Date(2017, 1, 1) .+ Day.(0:364) .+ Day(yearly_start)
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    plot_data = DataFrame(ds=days, Seasonality=seas[name])

    p = plot(plot_data, x=:ds, y=:Seasonality, Geom.line, color=[colorant"#0072B2"],
             Guide.xlabel("Day of year"), Guide.ylabel(name),
             Scale.x_continuous(format=:auto))

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]
        plot_data[!, Symbol(name * "_lower")] = lower
        plot_data[!, Symbol(name * "_upper")] = upper

        p = plot(plot_data,
                 layer(x=:ds, y=:Seasonality, Geom.line, color=[colorant"#0072B2"]),
                 layer(x=:ds, y=Symbol(name * "_lower"), y2=Symbol(name * "_upper"), 
                       Geom.ribbon, Theme(default_color=colorant"rgba(0, 114, 178, 0.2)")),
                 Guide.xlabel("Day of year"), Guide.ylabel(name),
                 Scale.x_continuous(format=:auto))
    end

    return p
end


function plot_yearly_plotly(m; uncertainty=true, yearly_start=0, name="yearly")
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = Date(2017, 1, 1) .+ Day.(0:364) .+ Day(yearly_start)
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    
    trace = scatter(; x=days, y=seas[name], mode="lines", name=name, line_color="rgb(0, 114, 178)")

    layout = Layout(xaxis_title="Day of year",
                    yaxis_title=name,
                    xaxis=attr(type="date"),
                    yaxis=attr())
    
    plot_data = [trace]

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]

        fill_trace = scatter(; x=days, y=lower, yaxis="y2",
                              mode="lines", line_width=0, 
                              showlegend=false, fill="toself",
                              fillcolor="rgba(0, 114, 178, 0.2)")
        line_trace = scatter(; x=days, y=upper, yaxis="y2",
                              mode="lines", line_width=0,
                              showlegend=false, fill="tonexty",
                              fillcolor="rgba(0, 114, 178, 0.2)")

        push!(plot_data, fill_trace, line_trace)
    end

    return plot(plot_data, layout)
end


function plot_seasonality_gadfly(m, name; uncertainty=true)
    # Compute seasonality from Jan 1 through a single period.
    start = DateTime(2017, 1, 1)
    period = m.seasonalities[name]["period"]
    end_date = start + Dates.Day(period)
    plot_points = 200
    days = range(start, end_date, length=plot_points)
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    plt = plot(df_y, x=:ds, y=seas[name],
               Geom.line, color=[colorant"#0072B2"],
               Guide.xlabel("Time"),
               Guide.ylabel(name),
               Guide.xticks(ticks=8))

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]

        plt = plot(df_y, x=:ds, y=seas[name], ymin=lower, ymax=upper,
                   Geom.line, Geom.ribbon(alpha=0.2),
                   color=[colorant"#0072B2"],
                   Guide.xlabel("Time"),
                   Guide.ylabel(name),
                   Guide.xticks(ticks=8))
    end

    return plt
end


function plot_seasonality_plotly(m, name; uncertainty=true)
    # Compute seasonality from Jan 1 through a single period.
    start = DateTime(2017, 1, 1)
    period = m.seasonalities[name]["period"]
    end_date = start + Dates.Day(period)
    plot_points = 200
    days = range(start, end_date, length=plot_points)
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    trace = scatter(; x=df_y.ds, y=seas[name], mode="lines", line_color="#0072B2")

    layout = Layout(; xaxis_title="Time", yaxis_title=name)

    if uncertainty && m.uncertainty_samples > 0
        lower = seas[name * "_lower"]
        upper = seas[name * "_upper"]

        fill_trace = scatter(; x=[df_y.ds; reverse(df_y.ds)],
                              y=[lower; reverse(upper)],
                              fill="toself", fillcolor="rgba(0, 114, 178, 0.2)",
                              line_color="rgba(255, 255, 255, 0)",
                              showlegend=false)

        plot_data = [trace, fill_trace]
    else
        plot_data = [trace]
    end

    plot_object = Plot(plot_data, layout)
    return plot_object
end


function set_y_as_percent_gadfly(guide)
    yticks = get(guide, :ticks)
    yticklabels = [@sprintf("%.4g%%", 100 * y) for y in yticks]
    return Guide.yticks(ticks=yticks, labels=yticklabels)
end


function set_y_as_percent_plotly(layout)
    yticks = layout["yaxis"]["tickvals"]
    yticklabels = [@sprintf("%.4g%%", 100 * y) for y in yticks]
    layout["yaxis"]["ticktext"] = yticklabels
    return layout
end


function add_changepoints_to_gadfly_plot(m, fcst; threshold=0.01, cp_color="red", cp_linestyle=:dash, trend=true)
    layers = []

    if trend
        push!(layers, layer(x=fcst.ds, y=fcst.trend, Geom.line, Theme(default_color=cp_color)))
    end

    deltas = vec(mean(m.params["delta"], dims=1))
    signif_changepoints = m.changepoints[abs.(deltas) .>= threshold]

    for cp in signif_changepoints
        push!(layers, layer(x=[cp, cp], y=[minimum(fcst.trend), maximum(fcst.trend)], Geom.line, Theme(default_color=cp_color, line_style=cp_linestyle)))
    end

    return plot(layers...)
end


function add_changepoints_to_plotly_plot(m, fcst; threshold=0.01, cp_color="red", cp_linestyle="dash", trend=true)
    traces = []

    if trend
        push!(traces, scatter(x=fcst.ds, y=fcst.trend, mode="lines", line=attr(color=cp_color)))
    end

    deltas = vec(mean(m.params["delta"], dims=1))
    signif_changepoints = m.changepoints[abs.(deltas) .>= threshold]

    for cp in signif_changepoints
        push!(traces, scatter(x=[cp, cp], y=[minimum(fcst.trend), maximum(fcst.trend)], mode="lines", line=attr(color=cp_color, dash=cp_linestyle)))
    end

    return plot(traces)
end


function plot_cross_validation_metric_gadfly(
        df_cv, metric, rolling_window=0.1;
        color="blue", point_color="gray"
    )
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=-1)
    df_h = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    plot(
        layer(x=df_none.horizon, y=df_none[:, Symbol(metric)], Geom.point, Theme(default_color=colorant(point_color), alphas=[0.1])),
        layer(x=df_h.horizon, y=df_h[:, Symbol(metric)], Geom.line, Theme(default_color=colorant(color))),
        Guide.xlabel("Horizon"),
        Guide.ylabel(metric)
    )
end

function plot_cross_validation_metric_plotly(
        df_cv, metric, rolling_window=0.1;
        color="blue", point_color="gray"
    )
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=-1)
    df_h = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    trace_points = scatter(
        x=df_none.horizon, y=df_none[:, Symbol(metric)],
        mode="markers", marker=attr(color=point_color, opacity=0.1)
    )

    trace_line = scatter(
        x=df_h.horizon, y=df_h[:, Symbol(metric)],
        mode="lines", line=attr(color=color)
    )

    layout = Layout(
        xaxis=attr(title="Horizon"),
        yaxis=attr(title=metric)
    )

    plot([trace_points, trace_line], layout)
end
