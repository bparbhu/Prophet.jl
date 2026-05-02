@testset "Seasonality and regressor API by backend" begin
    for backend in BACKENDS
        @testset "$(backend)" begin
            m = Prophet.ProphetModel(model_backend=backend)
            add_seasonality(m; name="monthly", period=30.5, fourier_order=3)
            @test model_backend(m) == backend
            @test haskey(m.seasonalities, "monthly")
            @test m.seasonalities["monthly"]["fourier_order"] == 3
            @test_throws ErrorException add_seasonality(m; name="bad", period=7, fourier_order=0)

            add_regressor(m; name="promo", prior_scale=2.0, standardize=false, mode="multiplicative")
            @test haskey(m.extra_regressors, "promo")
            @test m.extra_regressors["promo"]["mode"] == "multiplicative"
        end
    end

    dates = [Date(2012, 6, 1)]
    weekly = fourier_series(dates, 7, 3)
    @test size(weekly) == (1, 6)
    @test isapprox(
        weekly[1, :],
        [0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689];
        atol=1e-6,
    )

    features = make_seasonality_features(dates, 7, 3, "weekly")
    @test names(features) == [
        "weekly_delim_1", "weekly_delim_2", "weekly_delim_3",
        "weekly_delim_4", "weekly_delim_5", "weekly_delim_6",
    ]
end
