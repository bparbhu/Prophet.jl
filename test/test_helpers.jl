function example_daily(n::Int=120)
    ds = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))
    y = 10 .+ 0.25 .* collect(0:(n - 1)) .+ sin.(2pi .* collect(0:(n - 1)) ./ 7)
    return DataFrame(ds=collect(ds), y=Float64.(y))
end

const BACKENDS = (:stan, :turing, :neural_turing)

expected_fit_engine(backend::Symbol) = Dict(
    :stan => :stan_optimize,
    :turing => :turing_map,
    :neural_turing => :neural_turing_map,
)[backend]
