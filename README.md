# Prophet.jl

A Julia proof of concept for Prophet-style forecasting components.

This branch keeps the package runtime focused on Julia code:

- Stan-equivalent trend and likelihood utilities in `src/turing/prophet-turing.jl`.
- `Stan.jl` dependency and bundled Stan model access through `stan_model_file()`.
- A Turing model that mirrors the bundled Stan model's priors and likelihood.
- A Flux/Turing variant that keeps the Prophet terms equivalent and adds a Bayesian neural residual.
- Embedded holidays loaded from `data/generated_holidays.csv`, with no runtime holiday-package dependency.
- Plotting through Makie (`backend=:makie`) and Gadfly (`backend=:gadfly`).

## Python-Style Usage

Julia's package module is also named `Prophet`, so the closest equivalent to
Python's `m = Prophet()` is:

```julia
import Prophet

m = Prophet.Prophet()
Prophet.add_country_holidays(m; country_name="US")
Prophet.add_seasonality(m; name="monthly", period=30.5, fourier_order=3)
Prophet.fit(m, df)  # df is a DataFrame with ds and y columns
future = Prophet.make_future_dataframe(m; periods=365)
forecast = Prophet.predict(m, future)
fig = Prophet.plot(m, forecast; backend=:makie)
components = Prophet.plot_components(m, forecast; backend=:gadfly)
```

Select the modeling backend at construction time:

```julia
m_stan = Prophet.Prophet(model_backend=:stan)
m_turing = Prophet.Prophet(model_backend=:turing)
m_neural = Prophet.Prophet(model_backend=:neural_turing) # also accepts :flux_turing
```

The POC currently carries the backend choice through the public API while the
deterministic fit/predict facade remains lightweight for CI. The exported
`prophet` and `neural_prophet` Turing models are tested directly for their model
math.

If you prefer exported names:

```julia
using Prophet

m = ProphetModel()
fit(m, df)
forecast = predict(m, make_future_dataframe(m; periods=365))
```

## Diagnostics

The Python diagnostics flow is available with Julia keyword arguments:

```julia
using Dates

df_cv = Prophet.cross_validation(
    m;
    initial=Day(730),
    period=Day(180),
    horizon=Day(365),
)
df_p = Prophet.performance_metrics(df_cv)
```

Python Prophet supports `parallel="dask"` for distributed cross-validation. In
Julia, use Dagger for the same scheduler role:

```julia
using Dates

df_cv = Prophet.cross_validation(
    m;
    initial=Day(730),
    period=Day(180),
    horizon=Day(365),
    parallel=:dagger,
)
```

`parallel=:threads` is also available for a simple single-process threaded run.

## Holidays

Python Prophet depends on the Python `holidays` package. For this POC, the package reads the generated CSV directly:

```julia
using Prophet

make_holidays_df([2015, 2016], "US")
get_holiday_names("US")
supported_holiday_countries()
```

The maintenance script `src/scripts/generate_holidays_file.jl` normalizes an existing holidays CSV into the embedded format:

```bash
julia --project=. src/scripts/generate_holidays_file.jl
```

The input/output paths and year range can be overridden with:

- `PROPHET_HOLIDAYS_INPUT`
- `PROPHET_HOLIDAYS_OUTPUT`
- `PROPHET_HOLIDAYS_START_YEAR`
- `PROPHET_HOLIDAYS_END_YEAR`

## Plotting

```julia
plot_forecast(model, forecast; backend=:makie)
plot_forecast(model, forecast; backend=:gadfly)
plot_forecast_component(model, forecast, "trend"; backend=:makie)
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

GitHub Actions tests Julia `1.10` and the latest stable Julia release. CI also
downloads and builds CmdStan `2.37.0`, sets both `CMDSTAN` and
`JULIA_CMDSTAN_HOME`, and verifies the `stanc` version during the test run.
