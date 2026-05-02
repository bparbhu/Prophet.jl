@testset "Python models.py parity" begin
    @test Int(LINEAR) == 0
    @test Int(LOGISTIC) == 1
    @test Int(FLAT) == 2
    @test trend_indicator("linear") == 0
    @test trend_indicator("logistic") == 1
    @test trend_indicator("flat") == 2

    data = Dict{String,Any}(
        "T" => 3,
        "S" => 2,
        "K" => 1,
        "tau" => 0.05,
        "trend_indicator" => trend_indicator("linear"),
        "y" => [1.0, 2.0, 3.0],
        "t" => [0.0, 0.5, 1.0],
        "cap" => [0.0, 0.0, 0.0],
        "t_change" => [0.25, 0.75],
        "s_a" => [1.0],
        "s_m" => [0.0],
        "X" => reshape([1.0, 0.0, 1.0], 3, 1),
        "sigmas" => [10.0],
    )
    input = model_input_data(data)
    @test input.T == 3
    @test input.S == 2
    @test input.K == 1
    @test input.X == data["X"]
    @test model_input_data_dict(input)["t_change"] == data["t_change"]

    params = ModelParams(1.0, 0.2, [0.1, -0.1], [0.5], 0.01)
    params_dict = model_params_dict(params)
    @test params_dict["k"] == 1.0
    @test params_dict["delta"] == [0.1, -0.1]

    defaults = Dict{String,Any}(
        "k" => 1.0,
        "m" => 0.0,
        "sigma_obs" => 0.1,
        "delta" => zeros(2),
        "beta" => zeros(1),
    )
    custom = Dict{String,Any}(
        "k" => "bad",
        "m" => 2,
        "sigma_obs" => 0.2,
        "delta" => [0.1],
        "beta" => [0.3],
    )
    sanitized = sanitize_custom_inits(defaults, custom)
    @test sanitized["k"] == defaults["k"]
    @test sanitized["m"] == 2.0
    @test sanitized["sigma_obs"] == 0.2
    @test sanitized["delta"] == defaults["delta"]
    @test sanitized["beta"] == [0.3]

    parsed = stan_to_dict(["k", "m", "delta.1", "delta.2", "beta.1", "sigma_obs"], [1.0, 0.2, 0.1, -0.1, 0.5, 0.01])
    @test parsed["k"] == [1.0]
    @test parsed["m"] == [0.2]
    @test parsed["delta"] == [0.1 -0.1]
    @test parsed["beta"] == [0.5]
    @test parsed["sigma_obs"] == [0.01]
end
