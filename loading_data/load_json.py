import pandas


path = 'sample.json'

# loading the JSON file
dataframe = pandas.read_json(path, orient='index')
# dataframe = pandas.read_json(path, orient='columns')
# orient: split, records, index, columns, and values
# json_normalize(): helps convert semi-structured JSON data into a pandas DataFrame
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html

print(dataframe.head(2))    # returns the first n rows; -1 --> except the last one
print("****************")
print(dataframe.tail(2))     # last n rows
print("****************")
print(dataframe)
