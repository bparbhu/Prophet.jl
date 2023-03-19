using DataFrames, CSV
using Dates
import hdays

function get_holiday_names(country::AbstractString)
    years = 1995:2044

    try
        holiday_names = values(getfield(hdays.Holidays, Symbol(country))(years=years).holidays)
    catch e
        if isa(e, UndefVarError)
            error("Holidays in $country are not currently supported!")
        else
            rethrow()
        end
    end

    return Set(holiday_names)
end

function make_holidays_df(year_list::Vector{Int}, country::AbstractString, province::Union{Nothing, AbstractString}=nothing, state::Union{Nothing, AbstractString}=nothing)
    try
        holidays = getfield(hdays.Holidays, Symbol(country))(years=year_list, expand=false).holidays
    catch e
        if isa(e, UndefVarError)
            error("Holidays in $country are not currently supported!")
        else
            rethrow()
        end
    end

    holidays_df = DataFrame(ds=Date[], holiday=String[])
    for (date, holiday_list) in holidays
        for holiday in holiday_list
            push!(holidays_df, (date, holiday))
        end
    end

    return holidays_df
end