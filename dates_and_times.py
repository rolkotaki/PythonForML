import numpy as np
import pandas as pd


# Converting Strings to Dates

date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])
# Convert to datetimes
print([pd.to_datetime(date, format='%d-%m-%Y %I:%M %p', errors="coerce") for date in date_strings])
# errors="coerce": any problem that occurs will not raise an error, but will set the value causing the error to NaT
# format parameter to specify the exact format of the string


# Handling Time Zones

# Create datetime
date = pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
# OR
date = pd.Timestamp('2017-05-01 06:00:00')
# Set time zone
date_in_london = date.tz_localize('Europe/London')
print(date_in_london)
print(date_in_london.tz_convert('Africa/Abidjan'))

# pandas’ Series objects can apply tz_localize and tz_convert to every element
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
print(dates.dt.tz_localize('Africa/Abidjan'))


# Selecting Dates and Times

dataframe = pd.DataFrame()
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
# Select observations between two datetimes
print(dataframe[(dataframe['date'] > '2002-1-1 01:00:00') & (dataframe['date'] <= '2002-1-1 04:00:00')])


# Breaking Up Date Data into Multiple Features

dataframe = pd.DataFrame()
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')
# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute
dataframe['weekday_num'] = dataframe['date'].dt.weekday
print(dataframe.head(3))


# Calculating the Difference Between Dates

dataframe = pd.DataFrame()
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
dataframe['Travel_time'] = dataframe['Left'] - dataframe['Arrived']
print(dataframe)


# Creating a Lagged Feature

dataframe = pd.DataFrame()
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1, 2.2, 3.3, 4.4, 5.5]
# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)
print(dataframe)


# Using Rolling Time Windows

dates = pd.date_range("01/01/2010", periods=5, freq="M")
dataframe = pd.DataFrame(index=dates)
dataframe["Stock_Price"] = [1, 2, 3, 4, 5]
# Calculate rolling mean
dataframe = dataframe.rolling(window=2).mean()
print(dataframe)


# Handling Missing Data in Time Series

dates = pd.date_range("01/01/2010", periods=5, freq="M")
dataframe = pd.DataFrame(index=dates)
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]
# Interpolate missing values
print(dataframe.interpolate())
# Forward-fill - with the last known value
print(dataframe.ffill())
# Back-fill - with the latest known value
print(dataframe.bfill())

# to restrict the number and direction of interpolated values
print(dataframe.interpolate(limit=1, limit_direction="forward"))
# If the line between the two points is nonlinear, we can use interpolate’s method to specify the interpolation method:
print(dataframe.interpolate(method="quadratic"))
