import pandas
# have to install xlrd package


path = 'traininghistory.xlsx'

# loading the Excel file
dataframe = pandas.read_excel(path, sheet_name=0, header=0)
# sheet_name: sheet name string or zero-indexed position

print(dataframe.head(2))    # returns the first n rows; -1 --> except the last one
print("****************")
print(dataframe.tail(2))     # last n rows
print("****************")
# print(dataframe)
