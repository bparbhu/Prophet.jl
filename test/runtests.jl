using Test
using Dates
using DataFrames
import CairoMakie
import Gadfly
import Stan
using Prophet

include("test_helpers.jl")

@testset "Prophet.jl" begin
    include("constructor_tests.jl")
    include("test_models_tests.jl")
    include("stan_backend_tests.jl")
    include("forecaster_tests.jl")
    include("test_prophet_tests.jl")
    include("growth_tests.jl")
    include("seasonality_tests.jl")
    include("holidays_tests.jl")
    include("turing_tests.jl")
    include("neural_turing_tests.jl")
    include("diagnostics_tests.jl")
    include("test_diagnostics_tests.jl")
    include("test_serialize_tests.jl")
    include("test_utilities_tests.jl")
    include("plot_tests.jl")
    include("examples_tests.jl")
end
