using Dates
using LunarCalendar, HolidayBase
using Base.Iterators: product
using Base.Dates: Date
using PyCall
using Hijri
using IslamicDates
using AstroTime


const WEEKEND = HolidayBase.WEEKEND
const EASTER_ORTHODOX = pyimport("dateutil.easter").EASTER_ORTHODOX
const easter = pyimport("dateutil.easter").easter
const rd = pyimport("dateutil.relativedelta").relativedelta
const from_gregorian = pyimport("convertdate.islamic").from_gregorian
const to_gregorian = pyimport("convertdate.islamic").to_gregorian

mutable struct Indonesia <: HolidayBase.AbstractHolidayBase
    country::String
end

function Indonesia(; kwargs...)
    new("ID", kwargs...)
end

function HolidayBase._populate(holidays::Indonesia, year::Int)
    # New Year's Day
    if !holidays.observed && dayofweek(Date(year, 1, 1)) ∈ WEEKEND
        pass
    else
        holidays[Date(year, 1, 1)] = "New Year's Day"
    end

    # Chinese New Year/ Spring Festival
    name = "Chinese New Year"
    for offset in -1:1
        ds = LunarCalendar.Lunar2Solar(Lunar(year + offset, 1, 1)).to_date()
        if ds.year == year
            holidays[ds] = name
        end
    end

    # Day of Silence / Nyepi
    # Note:
    # This holiday is determined by Balinese calendar, which is not currently
    # available. Only hard coded version of this holiday from 2009 to 2019
    # is available.
    warning_msg = "We only support Nyepi holiday from 2009 to 2019"
    @warn warning_msg

    name = "Day of Silence/ Nyepi"
    if year == 2009
        holidays[Date(year, 3, 26)] = name
    elseif year == 2010
        holidays[Date(year, 3, 16)] = name
    elseif year == 2011
        holidays[Date(year, 3, 5)] = name
    elseif year == 2012
        holidays[Date(year, 3, 23)] = name
    elseif year == 2013
        holidays[Date(year, 3, 12)] = name
    elseif year == 2014
        holidays[Date(year, 3, 31)] = name
    elseif year == 2015
        holidays[Date(year, 3, 21)] = name
    elseif year == 2016
        holidays[Date(year, 3, 9)] = name
    elseif year == 2017
        holidays[Date(year, 3, 28)] = name
    elseif year == 2018
        holidays[Date(year, 3, 17)] = name
    elseif year == 2019
        holidays[Date(year, 3, 7)] = name
    else
        pass
    end

    # Ascension of the Prophet
    name = "Ascension of the Prophet"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 3, 17)[1]
        y, m, d = to_gregorian(islam_year, 7, 27)
        if y == year
            holidays[Date(y, m, d)] = name
        end
    end

   # Labor Day
   name = "Labor Day"
   holidays[Date(year, 5, 1)] = name

   # Ascension of Jesus Christ
   name = "Ascension of Jesus"
   for offset in -1:1
       ds = easter(year + offset) + rd(days=+39)
       if ds.year == year
           holidays[Date(ds)] = name
       end
   end

   # Buddha's Birthday
   name = "Buddha's Birthday"
   for offset in -1:1
       ds = LunarCalendar.Lunar2Solar(Lunar(year + offset, 4, 15)).to_date()
       if ds.year == year
           holidays[ds] = name
       end
   end

   # Pancasila Day, since 2017
   if year >= 2017
       name = "Pancasila Day"
       holidays[Date(year, 6, 1)] = name
   end

   # Eid al-Fitr
   name = "Eid al-Fitr"
   for offset in -1:1
       islam_year = from_gregorian(year + offset, 6, 15)[1]
       y1, m1, d1 = to_gregorian(islam_year, 10, 1)
       y2, m2, d2 = to_gregorian(islam_year, 10, 2)
       if y1 == year
           holidays[Date(y1, m1, d1)] = name
       end
       if y2 == year
           holidays[Date(y2, m2, d2)] = name
       end
   end

   # Independence Day
   name = "Independence Day"
   holidays[Date(year, 8, 17)] = name

   # Feast of the Sacrifice
   name = "Feast of the Sacrifice"
   for offset in -1:1
       islam_year = from_gregorian(year + offset, 8, 22)[1]
       y, m, d = to_gregorian(islam_year, 12, 10)
       if y == year
           holidays[Date(y, m, d)] = name
       end
   end

   # Islamic New Year
   name = "Islamic New Year"
   for offset in -1:1
       islam_year = from_gregorian(year + offset, 9, 11)[1]
       y, m, d = to_gregorian(islam_year + 1, 1, 1)
       if y == year
           holidays[Date(y, m, d)] = name
       end
   end

   # Birth of the Prophet
   name = "Birth of the Prophet"
   for offset in -1:1
       islam_year = from_gregorian(year + offset, 11, 20)[1]
       y, m, d = to_gregorian(islam_year + 1, 3, 12)
       if y == year
           holidays[Date(y, m, d)] = name
       end
   end

   # Christmas
   holidays[Date(year, 12, 25)] = "Christmas"
end


mutable struct ID <: Indonesia
end

mutable struct India <: Holidays.HolidayCalendar
    country::String
end

India() = India("IN")

Base.push!(dict::Dict{Date, String}, date::Date, name::String) = dict[date] = name

function Holidays.holidays(c::India, year::Integer)
    holidays = Dict{Date, String}()

    # Helper functions for adding holidays
    add_republic_day!(year, holidays) = push!(holidays, Date(year, 1, 26), "Republic Day")
    add_independence_day!(year, holidays) = push!(holidays, Date(year, 8, 15), "Independence Day")
    add_gandhi_jayanti!(year, holidays) = push!(holidays, Date(year, 10, 2), "Gandhi Jayanti")
    
    # Add National Holidays
    add_republic_day!(year, holidays)
    add_independence_day!(year, holidays)
    add_gandhi_jayanti!(year, holidays)

    # Add Hindu holidays
    add_diwali!(year, holidays) = begin
        diwali_dates = Dict(
            2010 => Date(2010, 11, 5),
            2011 => Date(2011, 10, 26),
            2012 => Date(2012, 11, 13),
            2013 => Date(2013, 11, 3),
            2014 => Date(2014, 10, 23),
            2015 => Date(2015, 11, 11),
            2016 => Date(2016, 10, 30),
            2017 => Date(2017, 10, 19),
            2018 => Date(2018, 11, 7),
            2019 => Date(2019, 10, 27),
            2020 => Date(2020, 11, 14),
            2021 => Date(2021, 11, 4),
            2022 => Date(2022, 10, 24),
            2023 => Date(2023, 10, 12),
            2024 => Date(2024, 11, 1),
            2025 => Date(2025, 10, 21),
            2026 => Date(2026, 11, 8),
            2027 => Date(2027, 10, 29),
            2028 => Date(2028, 10, 17),
            2029 => Date(2029, 11, 5),
            2030 => Date(2030, 10, 26)
        )
        if haskey(diwali_dates, year)
            push!(holidays, diwali_dates[year], "Diwali")
        end
    end
    
    add_holi!(year, holidays) = begin
        holi_dates = Dict(
            2010 => Date(2010, 2, 28),
            2011 => Date(2011, 3, 19),
            2012 => Date(2012, 3, 8),
            2013 => Date(2013, 3, 26),
            2014 => Date(2014, 3, 17),
            2015 => Date(2015, 3, 6),
            2016 => Date(2016, 3, 23),
            2017 => Date(2017, 3, 13),
            2018 => Date(2018, 3, 1),
            2019 => Date(2019, 3, 21),
            2020 => Date(2020, 3, 9),
            2021 => Date(2021, 3, 28),
            2022 => Date(2022, 3, 18),
            2023 => Date(2023, 3, 7),
            2024 => Date(2024, 3, 25),
            2025 => Date(2025, 3, 14),
            2026 => Date(2026, 3, 3),
            2027 => Date(2027, 3, 23),
            2028 => Date(2028, 3, 11),
            2029 => Date(2029, 3, 1),
            2030 => Date(2030, 3, 19)
        )
        if haskey(holi_dates, year)
            push!(holidays, holi_dates[year], "Holi")
        end
    end

    function add_islamic_holidays!(year, holidays)
        name = "Day of Ashura"
        for offset in -1:1
            islam_year = from_gregorian(year + offset, 10, 1)[1]
            y, m, d = to_gregorian(islam_year, 1, 10)
            if y == year
                holidays[Date(y, m, d)] = name
            end
        end
    
        name = "Mawlid"
        for offset in -1:1
            islam_year = from_gregorian(year + offset, 11, 20)[1]
            y, m, d = to_gregorian(islam_year, 3, 12)
            if y == year
                holidays[Date(y, m, d)] = name
            end
        end
    
        name = "Eid al-Fitr"
        for offset in -1:1
            islam_year = from_gregorian(year + offset, 6, 15)[1]
            y1, m1, d1 = to_gregorian(islam_year, 10, 1)
            y2, m2, d2 = to_gregorian(islam_year, 10, 2)
            if y1 == year
                holidays[Date(y1, m1, d1)] = name
            end
            if y2 == year
                holidays[Date(y2, m2, d2)] = name
            end
        end
    
        name = "Feast of the Sacrifice"
        for offset in -1:1
            islam_year = from_gregorian(year + offset, 8, 22)[1]
            y, m, d = to_gregorian(islam_year, 12, 10)
            if y == year
                holidays[Date(y, m, d)] = name
            end
        end
    end
    
    function add_christian_holidays!(year, holidays)
        holidays[Date(year, 1, 1)] = "New Year's Day"
    
        name = "Palm Sunday"
        for offset in -1:1
            ds = easter(year + offset) - Dates.Day(7)
            if ds.year == year
                holidays[ds] = name
            end
        end
    
        # ... (similar loops for Maundy Thursday, Good Friday, Easter Sunday, Feast of Pentecost)
    
        holidays[Date(year, 9, 5)] = "Fest of St. Theresa of Calcutta"
        holidays[Date(year, 9, 8)] = "Feast of the Blessed Virgin"
        holidays[Date(year, 11, 1)] = "All Saints Day"
        holidays[Date(year, 11, 2)] = "All Souls Day"
        holidays[Date(year, 12, 25)] = "Christmas Day"
        holidays[Date(year, 12, 26)] = "Boxing Day"
        holidays[Date(year, 12, 30)] = "Feast of Holy Family"
    end
    
    function india_holidays(year::Int)
        holidays = Dict{Date, String}()
    
        add_fixed_holidays!(year, holidays)
        add_diwali!(year, holidays)
        add_holi!(year, holidays)
        add_islamic_holidays!(year, holidays)
        add_christian_holidays!(year, holidays)
    
        return holidays
    end

    
function add_fixed_holidays_kg!(year, holidays)
    holidays[Date(year, 1, 1)] = "New Year's Day"
    holidays[Date(year, 1, 7)] = "Orthodox Christmas Day"
    holidays[Date(year, 2, 23)] = "Fatherland Defender's Day"
    holidays[Date(year, 3, 8)] = "International Women's Day"
    holidays[Date(year, 3, 21)] = "Nooruz Mairamy"
    holidays[Date(year, 4, 7)] = "Day of the People's April Revolution"
    holidays[Date(year, 5, 1)] = "Spring and Labour Day"
    holidays[Date(year, 5, 5)] = "Constitution Day"
    holidays[Date(year, 5, 9)] = "Victory Day"
    holidays[Date(year, 6, 1)] = "Russia Day"
    holidays[Date(year, 8, 31)] = "Independence Day"
    holidays[Date(year, 11, 7)] = "Day 1 of History and Commemoration of Ancestors"
    holidays[Date(year, 11, 8)] = "Day 2 of History and Commemoration of Ancestors"
    holidays[Date(year, 12, 31)] = "New Year's Eve"
end

function add_islamic_holidays_kg!(year, holidays)
    name = "Eid al-Fitr"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 6, 15)[1]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        if y1 == year
            holidays[Date(y1, m1, d1)] = name
        end
        if y2 == year
            holidays[Date(y2, m2, d2)] = name
        end
    end

    name = "Feast of the Sacrifice"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 8, 22)[1]
        y, m, d = to_gregorian(islam_year, 12, 10)
        if y == year
            holidays[Date(y, m, d)] = name
        end
    end
end

function kyrgyzstan_holidays(year::Int)
    holidays = Dict{Date, String}()

    add_fixed_holidays_kg!(year, holidays)
    add_islamic_holidays_kg!(year, holidays)

    return holidays
end


using Dates

abstract type HolidayBase end

mutable struct Thailand <: HolidayBase
    country::String
    holidays::Dict{Date, String}
    
    function Thailand(; kwargs...)
        self = new()
        self.country = "TH"
        self.holidays = Dict{Date, String}()
        return self
    end
end

function populate(self::Thailand, year::Int)
    # New Year's Day
    name = "New Year's Day"
    self.holidays[Date(year, 1, 1)] = name

    # Magha Pujab
    # Note:
    # This holiday is determined by Buddhist calendar, which is not currently
    # available. Only hard coded version of this holiday from 2016 to 2019
    # is available.

    name = "Magha Pujab/Makha Bucha"
    if year == 2016
        self.holidays[Date(year, 2, 22)] = name
    elseif year == 2017
        self.holidays[Date(year, 2, 11)] = name
    elseif year == 2018
        self.holidays[Date(year, 3, 1)] = name
    elseif year == 2019
        self.holidays[Date(year, 2, 19)] = name
    else
        # do nothing
    end

    # Chakri Memorial Day
    name = "Chakri Memorial Day"
    april_6 = Dates.dayofweek(Date(year, 4, 6))
    if april_6 == 5
        self.holidays[Date(year, 4, 6 + 2)] = name
    elseif april_6 == 6
        self.holidays[Date(year, 4, 6 + 1)] = name
    else
        self.holidays[Date(year, 4, 6)] = name
    end

    # Songkran Festival
    name = "Songkran Festival"
    self.holidays[Date(year, 4, 14)] = name

    # Royal Ploughing Ceremony
    # arbitrary day in May
    # Buddha's Birthday
    name = "Buddha's Birthday"
    for offset in range(-1, 2, 1)
        lun_date = Chinese(year+offset, 4, 15)
        ds = from_chinese(lun_date, UT1())
        if ds.year == year
            self.holidays[Date(ds)] = name
        end
    end
    # Coronation Day, removed in 2017
    name = "Coronation Day"
    if year < 2017
        self.holidays[Date(year, 5, 5)] = name
    end

    # King Maha Vajiralongkorn's Birthday
    name = "King Maha Vajiralongkorn's Birthday"
    self.holidays[Date(year, 7, 28)] = name

    # Asalha Puja
    # This is also a Buddha holiday, and we only implement
    # the hard coded version from 2006 to 2025
    # reference:
    # http://www.when-is.com/asalha_puja.asp
    warning_msg = "We only support Asalha Puja holiday from 2006 to 2025"
    @warn warning_msg
    name = "Asalha Puja"
    if year == 2006
        self.holidays[Date(year, 7, 11)] = name
    elseif year == 2007
        self.holidays[Date(year, 6, 30)] = name
    elseif year == 2008
        self.holidays[Date(year, 7, 18)] = name
    elseif year == 2009
        self.holidays[Date(year, 7, 7)] = name
    elseif year == 2010
        self.holidays[Date(year, 7, 25)] = name
    elseif year == 2011
        self.holidays[Date(year, 7, 15)] = name
    elseif year == 2012
        self.holidays[Date(year, 8, 2)] = name
    elseif year == 2013
        self.holidays[Date(year, 7, 30)] = name
    elseif year == 2014
        self.holidays[Date(year, 7, 13)] = name
    elseif year == 2015
        self.holidays[Date(year, 7, 30)] = name
    elseif year == 2016
        self.holidays[Date(year, 7, 15)] = name
    elseif year == 2017
        self.holidays[Date(year, 7, 9)] = name
    elseif year == 2018
        self.holidays[Date(year, 7, 29)] = name
    elseif year == 2019
        self.holidays[Date(year, 7, 16)] = name
    elseif year == 2020
        self.holidays[Date(year, 7, 5)] = name
    elseif year == 2021
        self.holidays[Date(year, 7, 24)] = name
    elseif year == 2022
        self.holidays[Date(year, 7, 13)] = name
    elseif year == 2023
        self.holidays[Date(year, 7, 3)] = name
    elseif year == 2024
        self.holidays[Date(year, 7, 21)] = name
    elseif year == 2025
        self.holidays[Date(year, 7, 10)] = name
    else
        pass
    end

    # Beginning of Vassa
    warning_msg = "We only support Vassa holiday from 2006 to 2020"
    @warn warning_msg
    name = "Beginning of Vassa"
    if year == 2006
        self.holidays[Date(year, 7, 12)] = name
    elseif year == 2007
        self.holidays[Date(year, 7, 31)] = name
    elseif year == 2008
        self.holidays[Date(year, 7, 19)] = name
    elseif year == 2009
        self.holidays[Date(year, 7, 8)] = name
    elseif year == 2010
        self.holidays[Date(year, 7, 27)] = name
    elseif year == 2011
        self.holidays[Date(year, 7, 16)] = name
    elseif year == 2012
        self.holidays[Date(year, 8, 3)] = name
    elseif year == 2013
        self.holidays[Date(year, 7, 23)] = name
    elseif year == 2014
        self.holidays[Date(year, 7, 13)] = name
    elseif year == 2015
        self.holidays[Date(year, 8, 1)] = name
    elseif year == 2016
        self.holidays[Date(year, 7, 20)] = name
    elseif year == 2017
        self.holidays[Date(year, 7, 9)] = name
    elseif year == 2018
        self.holidays[Date(year, 7, 28)] = name
    elseif year == 2019
        self.holidays[Date(year, 7, 17)] = name
    elseif year == 2020
        self.holidays[Date(year, 7, 6)] = name
    else
        pass
    end

    # The Queen Sirikit's Birthday
    name = "The Queen Sirikit's Birthday"
    self.holidays[Date(year, 8, 12)] = name

    # Anniversary for the Death of King Bhumibol Adulyadej
    name = "Anniversary for the Death of King Bhumibol Adulyadej"
    self.holidays[Date(year, 10, 13)] = name

    # King Chulalongkorn Day
    name = "King Chulalongkorn Day"
    self.holidays[Date(year, 10, 23)] = name

    # King Bhumibol Adulyadej's Birthday Anniversary
    name = "King Bhumibol Adulyadej's Birthday Anniversary"
    self.holidays[Date(year, 12, 5)] = name

    # Constitution Day
    name = "Constitution Day"
    self.holidays[Date(year, 12, 10)] = name

    # New Year's Eve
    name = "New Year's Eve"
    self.holidays[Date(year, 12, 31)] = name
end

struct TH <: Thailand
function TH(; kwargs...)
    self = new()
    self.country = "TH"
    self.holidays = Dict{Date, String}()
    return self
end
end

# ------------ Holidays in Philippines ---------------------
mutable struct Philippines <: HolidayBase
    holidays::Dict{Date, String}
    country::String

    function Philippines(;kwargs...)
        self = new()
        self.holidays = Dict{Date, String}()
        self.country = "PH"
        HolidayBase.init(self; kwargs...)
        return self
    end
end

function HolidayBase._populate(self::Philippines, year::Int)
    # New Year's Day
    name = "New Year's Day"
    self.holidays[Date(year, 1, 1)] = name

    # Maundy Thursday
    name = "Maundy Thursday"
    for offset in -1:1
        ds = easter(year + offset) - Dates.Day(3)
        if year == ds.year
            self.holidays[ds] = name
        end
    end

    # Good Friday
    name = "Good Friday"
    for offset in -1:1
        ds = easter(year + offset) - Dates.Day(2)
        if year == ds.year
            self.holidays[ds] = name
        end
    end

    # Day of Valor
    name = "Day of Valor"
    self.holidays[Date(year, 4, 9)] = name

    # Labor Day
    name = "Labor Day"
    self.holidays[Date(year, 5, 1)] = name

    # Independence Day
    name = "Independence Day"
    self.holidays[Date(year, 6, 12)] = name

    # Eid al-Fitr
    name = "Eid al-Fitr"
    for offset in -1:1
        islam_year, _, _ = from_gregorian(year + offset, 6, 15)
        y, m, d = to_gregorian(islam_year, 10, 1)
        ds = Date(y, m, d) - Dates.Day(1)
        if year == ds.year
            self.holidays[ds] = name
        end
    end

    # Eid al-Adha, i.e., Feast of the Sacrifice
    name = "Feast of the Sacrifice"
    for offset in -1:1
        islam_year, _, _ = from_gregorian(year + offset, 8, 22)
        y, m, d = to_gregorian(islam_year, 12, 10)
        if year == y
            self.holidays[Date(y, m, d)] = name
        end
    end

    # National Heroes' Day
    name = "National Heroes' Day"
    self.holidays[Date(year, 8, 27)] = name

    # Bonifacio Day
    name = "Bonifacio Day"
    self.holidays[Date(year, 11, 30)] = name

    # Christmas Day
    name = "Christmas Day"
    self.holidays[Date(year, 12, 25)] = name
    name = "Rizal Day"
    self.holidays[Date(year, 12, 30)] = name
end

mutable struct PH <: Philippines
    holidays::Dict{Date, String}
    country::String

    function PH(;kwargs...)
        self = new()
        self.holidays = Dict{Date, String}()
        self.country = "PH"
        HolidayBase.init(self; kwargs...)
        return self
    end
end

abstract type HolidayBase end

mutable struct Pakistan <: HolidayBase
    holidays::Dict{Date, String}
    country::String

    function Pakistan(;kwargs...)
        self = new()
        self.holidays = Dict{Date, String}()
        self.country = "PK"
        HolidayBase.init(self; kwargs...)
        return self
    end
end

function HolidayBase.init(self::Pakistan; kwargs...)
    year = get(kwargs, :year, Dates.year(Dates.today()))
    populate(self, year)
end

function populate(self::Pakistan, year)
    # Kashmir Solidarity Day
    name = "Kashmir Solidarity Day"
    self.holidays[Date(year, 2, 5)] = name

    # Pakistan Day
    name = "Pakistan Day"
    self.holidays[Date(year, 3, 23)] = name

    # Labor Day
    name = "Labor Day"
    self.holidays[Date(year, 5, 1)] = name

    # Independence Day
    name = "Independence Day"
    self.holidays[Date(year, 8, 14)] = name

    # Iqbal Day
    name = "Iqbal Day"
    self.holidays[Date(year, 11, 9)] = name

    # Christmas Day
    # Also birthday of PK founder
    name = "Christmas Day"
    self.holidays[Date(year, 12, 25)] = name

    # Eid al-Adha, i.e., Feast of the Sacrifice
    name = "Feast of the Sacrifice"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 8, 22)[1]
        y1, m1, d1 = to_gregorian(islam_year, 12, 10)
        y2, m2, d2 = to_gregorian(islam_year, 12, 11)
        y3, m3, d3 = to_gregorian(islam_year, 12, 12)
        if y1 == year
            self.holidays[Date(y1, m1, d1)] = name
        end
        if y2 == year
            self.holidays[Date(y2, m2, d2)] = name
        end
        if y3 == year
            self.holidays[Date(y3, m3, d3)] = name
        end
    end

    # Eid al-Fitr
    name = "Eid al-Fitr"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 6, 15)[1]
        y1, m1, d1 = to_gregorian(islam_year, 10, 1)
        y2, m2, d2 = to_gregorian(islam_year, 10, 2)
        y3, m3, d3 = to_gregorian(islam_year, 10, 3)
        if y1 == year
            self.holidays[Date(y1, m1, d1)] = name
        end
        if y2 == year
            self.holidays[Date(y2, m2, d2)] = name
        end
        if y3 == year
            self.holidays[Date(y3, m3, d3)] = name
        end
    end

    # Mawlid, Birth of the Prophet
    # 12th day of 3rd Islamic month
    name = "Mawlid"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 11, 20)[1]
        y, m, d = to_gregorian(islam_year, 3, 12)
        if y == year
            self.holidays[Date(y, m, d)] = name
        end
    end

    # Day of Ashura
    # 10th and 11th days of 1st Islamic month
    name = "Day of Ashura"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 10, 1)[1]
        y1, m1, d1 = to_gregorian(islam_year, 1, 10)
        y2, m2, d2 = to_gregorian(islam_year, 1, 11)
        if y1 == year
            self.holidays[Date(y1, m1, d1)] = name
        end
        if y2 == year
            self.holidays[Date(y2, m2, d2)] = name
        end
    end

    # Shab e Mairaj
    name = "Shab e Mairaj"
    for offset in -1:1
        islam_year = from_gregorian(year + offset, 4, 13)[1]
        y, m, d = to_gregorian(islam_year, 7, 27)
        if y == year
            self.holidays[Date(y, m, d)] = name
        end
    end

    # Defence Day
    name = "Defence Day"
    self.holidays[Date(year, 9, 6)] = name

    # Death Anniversary of Quaid-e-Azam
    name = "Death Anniversary of Quaid-e-Azam"
    self.holidays[Date(year, 9, 11)] = name
end

mutable struct PK <: Pakistan
    function PK(; kwargs...)
        new()
        Pakistan(; kwargs...)
    end
end

mutable struct Russia <: HolidayBase
    country::String
    holidays::Dict{Date, String}

    function Russia(; kwargs...)
        self = new()
        self.country = "RU"
        self.holidays = Dict{Date, String}()
        _populate(self, get(kwargs, :year, Dates.year(today())))
        return self
    end
end

function _populate(self::Russia, year::Int)
    # New Year's Day
    name = "New Year's Day"
    self.holidays[Date(year, 1, 1)] = name

    # Orthodox Christmas day
    name = "Orthodox Christmas Day"
    self.holidays[Date(year, 1, 7)] = name

    # Dec. 25 Christmas Day
    name = "Christmas Day"
    self.holidays[Date(year, 12, 25)] = name

    # Defender of the Fatherland Day
    name = "Defender of the Fatherland Day"
    self.holidays[Date(year, 2, 23)] = name

    # International Women's Day
    name = "International Women's Day"
    self.holidays[Date(year, 3, 8)] = name

    # National Flag Day
    name = "National Flag Day"
    self.holidays[Date(year, 8, 22)] = name

    # Spring and Labour Day
    name = "Spring and Labour Day"
    self.holidays[Date(year, 5, 1)] = name

    # Victory Day
    name = "Victory Day"
    self.holidays[Date(year, 5, 9)] = name

    # Russia Day
    name = "Russia Day"
    self.holidays[Date(year, 6, 12)] = name

    # Unity Day
    name = "Unity Day"
    self.holidays[Date(year, 11, 4)] = name
end

mutable struct RU <: Russia
    function RU(; kwargs...)
        new()
        Russia(; kwargs...)
    end
end

mutable struct Belarus <: HolidayBase
    country::String
    holidays::Dict{Date, String}

    function Belarus(; kwargs...)
        self = new()
        self.country = "BY"
        self.holidays = Dict{Date, String}()
        _populate(self, get(kwargs, :year, Dates.year(today())))
        return self
    end
end

function _populate(self::Belarus, year::Int)
    # New Year's Day
    name = "New Year's Day"
    self.holidays[Date(year, 1, 1)] = name

    # Orthodox Christmas day
    name = "Orthodox Christmas Day"
    self.holidays[Date(year, 1, 7)] = name

    # International Women's Day
    name = "International Women's Day"
    self.holidays[Date(year, 3, 8)] = name

    # Commemoration Day
    name = "Commemoration Day"
    easter_date = easter(year, EASTER_ORTHODOX)
    self.holidays[easter_date + Day(9)] = name

    # Spring and Labour Day
    name = "Spring and Labour Day"
    self.holidays[Date(year, 5, 1)] = name

    # Victory Day
    name = "Victory Day"
    self.holidays[Date(year, 5, 9)] = name

    # Independence Day
    name = "Independence Day"
    self.holidays[Date(year, 7, 3)] = name

    # October Revolution Day
    name = "October Revolution Day"
    self.holidays[Date(year, 11, 7)] = name

    # Dec. 25 Christmas Day
    name = "Christmas Day"
    self.holidays[Date(year, 12, 25)] = name
end

mutable struct BY <: Belarus
    function BY(; kwargs...)
        new()
        Belarus(; kwargs...)
    end
end


using Dates

mutable struct Georgia <: HolidayBase
    country::String
    holidays::Dict{Date, String}

    function Georgia(; kwargs...)
        self = new()
        self.country = "GE"
        self.holidays = Dict{Date, String}()
        _populate(self, get(kwargs, :year, Dates.year(today())))
        return self
    end
end

function _populate(self::Georgia, year::Int)
    # New Year's Day
    name = "New Year's Day"
    self.holidays[Date(year, 1, 1)] = name

    # Second day of the New Year
    name = "Second day of the New Year"
    self.holidays[Date(year, 1, 2)] = name

    # Orthodox Christmas
    name = "Orthodox Christmas"
    self.holidays[Date(year, 1, 7)] = name

    # Baptism Day of our Lord Jesus Christ
    name = "Baptism Day of our Lord Jesus Christ"
    self.holidays[Date(year, 1, 19)] = name

    # Mother's Day
    name = "Mother's Day"
    self.holidays[Date(year, 3, 3)] = name

    # International Women's Day
    name = "International Women's Day"
    self.holidays[Date(year, 3, 8)] = name

    # Orthodox Good Friday
    name = "Good Friday"
    easter_date = easter(year, EASTER_ORTHODOX)
    self.holidays[easter_date - Day(2)] = name

    # Orthodox Holy Saturday
    name = "Great Saturday"
    self.holidays[easter_date - Day(1)] = name

    # Orthodox Easter Sunday
    name = "Easter Sunday"
    self.holidays[easter_date] = name

    # Orthodox Easter Monday
    name = "Easter Monday"
    self.holidays[easter_date + Day(1)] = name

    # National Unity Day
    name = "National Unity Day"
    self.holidays[Date(year, 4, 9)] = name

    # Victory Day
    name = "Victory Day"
    self.holidays[Date(year, 5, 9)] = name

    # Saint Andrew the First-Called Day
    name = "Saint Andrew the First-Called Day"
    self.holidays[Date(year, 5, 12)] = name

    # Independence Day
    name = "Independence Day"
    self.holidays[Date(year, 5, 26)] = name

    # Saint Mary's Day
    name = "Saint Mary's Day"
    self.holidays[Date(year, 8, 28)] = name

    # Day of Svetitskhoveli Cathedral
    name = "Day of Svetitskhoveli Cathedral"
    self.holidays[Date(year, 10, 14)] = name

    # Saint George's Day
    name = "Saint George's Day"
    self.holidays[Date(year, 12, 23)] = name
end

mutable struct GE <: Georgia
    function GE(; kwargs...)
        new()
        Georgia(; kwargs...)
    end
end

