@testset "Python test_utilities.py parity by backend" begin
    @testset "warm start parameters" begin
        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                m = Prophet.ProphetModel(model_backend=backend, n_changepoints=4)
                fit(m, example_daily(35))
                warm = warm_start_params(m)
                @test Set(keys(warm)) == Set(["k", "m", "sigma_obs", "delta", "beta"])
                @test warm["k"] == m.params["k"]
                @test warm["m"] == m.params["m"]
                @test warm["sigma_obs"] == m.params["sigma_obs"]
                @test length(warm["delta"]) == length(m.params["delta"])
                @test length(warm["beta"]) == length(m.params["beta"])
            end
        end
    end

    @testset "regressor coefficients" begin
        @test_throws ErrorException regressor_coefficients(Prophet.ProphetModel(model_backend=:turing))

        for backend in FAST_BACKENDS
            @testset "$(backend)" begin
                df = example_daily(40)
                df.promo = Float64.(mod.(collect(1:nrow(df)), 2))
                df.price = Float64.(collect(1:nrow(df)))
                m = Prophet.ProphetModel(model_backend=backend)
                add_regressor(m; name="promo", standardize=false, mode="additive")
                add_regressor(m; name="price", standardize=true, mode="multiplicative")
                fit(m, df)

                @test regressor_index(m, "promo") > 0
                @test regressor_index(m, "price") > 0

                coefs = regressor_coefficients(m)
                @test names(coefs) == ["regressor", "regressor_mode", "center", "coef_lower", "coef", "coef_upper"]
                @test Set(coefs.regressor) == Set(["promo", "price"])
                @test Set(coefs.regressor_mode) == Set(["additive", "multiplicative"])
                @test all(coefs.coef_lower .<= coefs.coef)
                @test all(coefs.coef .<= coefs.coef_upper)
                @test coefs[coefs.regressor .== "promo", :center][1] == 0.0
                @test coefs[coefs.regressor .== "price", :center][1] != 0.0
            end
        end
    end
end
