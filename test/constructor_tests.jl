@testset "Prophet-style constructor and validation" begin
    m = Prophet.ProphetModel()
    @test m.growth == "linear"
    @test model_backend(m) == :stan
    @test m.n_changepoints == 25
    @test m.seasonality_mode == "additive"
    @test m.holidays_mode == "additive"

    @test_throws ErrorException Prophet.ProphetModel(growth="constant")
    @test_throws ErrorException Prophet.ProphetModel(changepoint_range=-0.1)
    @test_throws ErrorException Prophet.ProphetModel(changepoint_range=2.0)
    @test_throws ErrorException Prophet.ProphetModel(seasonality_mode="bad")
    @test_throws ErrorException Prophet.ProphetModel(model_backend=:bad)

    turing_model = Prophet.ProphetModel(model_backend=:turing)
    @test model_backend(turing_model) == :turing
    neural_model = Prophet.ProphetModel(model_backend=:flux_turing)
    @test model_backend(neural_model) == :neural_turing
    @test set_model_backend!(neural_model, :stan) === neural_model
    @test model_backend(neural_model) == :stan
end
