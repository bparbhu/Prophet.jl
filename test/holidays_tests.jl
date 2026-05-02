@testset "Embedded holidays and holiday features" begin
    countries = supported_holiday_countries()
    @test "US" in countries

    us = make_holidays_df([2015], "US")
    @test names(us) == ["ds", "holiday"]
    @test Date(2015, 1, 1) in us.ds
    @test "New Year's Day" in get_holiday_names("US")
    @test_throws ErrorException make_holidays_df([2015], "NO_SUCH_COUNTRY")

    m = Prophet.ProphetModel(holidays=DataFrame(
        holiday=["launch"],
        ds=[Date(2020, 1, 10)],
        lower_window=[-1],
        upper_window=[1],
        prior_scale=[4.0],
    ))
    holidays = construct_holiday_dataframe(m, Date(2020, 1, 8):Day(1):Date(2020, 1, 12))
    features, prior_scales, names_used = make_holiday_features(
        m, Date(2020, 1, 8):Day(1):Date(2020, 1, 12), holidays
    )
    @test names(features) == ["launch_delim_+0", "launch_delim_+1", "launch_delim_-1"]
    @test prior_scales == [4.0, 4.0, 4.0]
    @test names_used == ["launch"]
    @test sum(Matrix(features)) == 3.0

    for backend in BACKENDS
        m2 = Prophet.ProphetModel(model_backend=backend)
        add_country_holidays(m2; country_name="US")
        @test model_backend(m2) == backend
        @test m2.country_holidays == "US"
    end
end
