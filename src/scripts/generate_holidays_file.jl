#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Unicode

const DEFAULT_START_YEAR = 1995
const DEFAULT_END_YEAR = 2044
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_INPUT = joinpath(REPO_ROOT, "data", "generated_holidays.csv")
const DEFAULT_OUTPUT = joinpath(REPO_ROOT, "data", "generated_holidays.csv")

function ascii_name(name)
    stripped = strip(Unicode.normalize(String(name), stripmark=true))
    ascii = String([c for c in stripped if Int(c) <= 0x7f])
    return isempty(ascii) ? "FAILED_TO_PARSE" : ascii
end

function normalize_holidays_file(; input=DEFAULT_INPUT, output=DEFAULT_OUTPUT,
                                 start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR)
    df = CSV.read(input, DataFrame)
    required = ["ds", "holiday", "country"]
    missing_cols = setdiff(required, names(df))
    isempty(missing_cols) || error("Missing required holiday columns: $(join(missing_cols, ", "))")

    df.ds = Date.(df.ds)
    df.country = uppercase.(String.(df.country))
    df.holiday = ascii_name.(df.holiday)
    df.year = year.(df.ds)
    df = df[(start_year .<= df.year) .& (df.year .<= end_year) .& (df.holiday .!= "FAILED_TO_PARSE"), :]
    unique!(df)
    sort!(df, [:country, :ds, :holiday])

    CSV.write(output, df)
    return df
end

function main(args=ARGS)
    input = get(ENV, "PROPHET_HOLIDAYS_INPUT", DEFAULT_INPUT)
    output = get(ENV, "PROPHET_HOLIDAYS_OUTPUT", DEFAULT_OUTPUT)
    start_year = parse(Int, get(ENV, "PROPHET_HOLIDAYS_START_YEAR", string(DEFAULT_START_YEAR)))
    end_year = parse(Int, get(ENV, "PROPHET_HOLIDAYS_END_YEAR", string(DEFAULT_END_YEAR)))

    if "--help" in args || "-h" in args
        println("Regenerate/normalize data/generated_holidays.csv from an existing CSV source.")
        println("Override paths with PROPHET_HOLIDAYS_INPUT and PROPHET_HOLIDAYS_OUTPUT.")
        println("Override year range with PROPHET_HOLIDAYS_START_YEAR and PROPHET_HOLIDAYS_END_YEAR.")
        return nothing
    end

    df = normalize_holidays_file(
        input=input, output=output, start_year=start_year, end_year=end_year
    )
    println("Wrote $(nrow(df)) holidays to $output")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
