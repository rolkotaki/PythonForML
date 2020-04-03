import pandas


path = 'fruits.csv'

# loading the CSV file
dataframe = pandas.read_csv(filepath_or_buffer=path,
                            sep=',',
                            skip_blank_lines=True)  # header=None
# It has many parameters!

print(dataframe.head(2))    # returns the first n rows; -1 --> except the last one
print("****************")
print(dataframe.tail(1))     # last n rows
print("****************")
print(dataframe)
