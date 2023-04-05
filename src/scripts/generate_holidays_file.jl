using DataFrames
using Dates
using CSV

import .hdays
import holidays
import inspect
import unicodedata

function utf8_to_ascii(text)
    ascii_text = (
        unicodedata.normalize("NFD", text)
        |> x -> PyCall.encode(x, "ascii", "ignore")
        |> x -> PyCall.decode(x, "ascii")
        |> x -> strip(x)
    )

    if sum(1 for x in ascii_text if x ∉ [' ', '(', ')', ',']) == 0
        return "FAILED_TO_PARSE"
    else
        return ascii_text
    end
end

function generate_holidays_file()
    year_list = collect(1995:2044)
    all_holidays = []

    class_to_exclude = Set(["rd", "BY", "BG", "JP", "RS", "UA", "KR"])

    class_list2 = inspect.getmembers(hdays, inspect.isclass)
    country_set = Set([name for name in first.(class_list2) if length(name) == 2])
    class_list1 = inspect.getmembers(holidays, inspect.isclass)
    country_set1 = Set([name for name in first.(class_list1) if length(name) == 2])
    union!(country_set, country_set1)
    setdiff!(country_set, class_to_exclude)

    for country in country_set
        df = hdays2.make_holidays_df(year_list=year_list, country=country) |> DataFrame
        df[!, :country] .= country
        push!(all_holidays, df)
    end

    generated_holidays = vcat(all_holidays...)
    generated_holidays[!, :year] = Dates.year.(generated_holidays[!, :ds])
    sort!(generated_holidays, [:country, :ds, :holiday])

    generated_holidays[!, :holiday] = utf8_to_ascii.(generated_holidays[!, :holiday])
    failed_countries = unique(generated_holidays[generated_holidays.holiday .== "FAILED_TO_PARSE", :country])
    
    if !isempty(failed_countries)
        println("Failed to convert UTF-8 holidays for:")
        println(join(failed_countries, "\n"))
    end
    
    @assert "FAILED_TO_PARSE" ∉ unique(generated_holidays[!, :holiday])
    CSV.write("../R/data-raw/generated_holidays.csv", generated_holidays)
end

generate_holidays_file()
