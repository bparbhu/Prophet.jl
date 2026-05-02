using CSV
using DataFrames
using Dates

const HOLIDAYS_FILE = normpath(joinpath(@__DIR__, "..", "data", "generated_holidays.csv"))

function _holiday_table()
    isfile(HOLIDAYS_FILE) || error("Generated holiday file not found at $HOLIDAYS_FILE")
    df = CSV.read(HOLIDAYS_FILE, DataFrame)
    if !("ds" in names(df)) || !("holiday" in names(df)) || !("country" in names(df))
        error("Generated holiday file must contain ds, holiday, and country columns.")
    end
    df.ds = Date.(df.ds)
    return df
end

"""
    supported_holiday_countries()

Return the country codes available in the embedded generated holiday table.
"""
function supported_holiday_countries()
    return sort!(unique(String.(dropmissing(_holiday_table().country))))
end

"""
    get_holiday_names(country)

Return the set of known holiday names for a country code from the embedded generated
holiday table. This intentionally has no runtime dependency on a holiday package.
"""
function get_holiday_names(country::AbstractString)
    code = uppercase(country)
    df = _holiday_table()
    rows = df[df.country .== code, :]
    isempty(rows) && error("Holidays in $code are not currently supported.")
    return Set(String.(rows.holiday))
end

"""
    make_holidays_df(year_list, country; province=nothing, state=nothing)

Build a Prophet-compatible holiday DataFrame with `ds` and `holiday` columns from
the embedded generated holiday table.
"""
function make_holidays_df(
    year_list::AbstractVector{<:Integer},
    country::AbstractString;
    province::Union{Nothing,AbstractString}=nothing,
    state::Union{Nothing,AbstractString}=nothing,
)
    if province !== nothing || state !== nothing
        error("Subdivision holidays are not available in the embedded holiday table.")
    end

    code = uppercase(country)
    years = Set(Int.(year_list))
    df = _holiday_table()
    rows = df[(df.country .== code) .& in.(year.(df.ds), Ref(years)), [:ds, :holiday]]
    isempty(rows) && error("No embedded holidays found for $code in the requested years.")
    sort!(rows, [:ds, :holiday])
    return rows
end

make_holidays_df(year_list::AbstractVector{<:Integer}, country::AbstractString, province, state) =
    make_holidays_df(year_list, country; province=province, state=state)
