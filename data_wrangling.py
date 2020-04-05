import pandas
import numpy as np

url = 'https://tinyurl.com/titanic-csv'
dataframe = pandas.read_csv(url)

print(dataframe.iloc[0])
print(dataframe.iloc[1:3])

print(dataframe.shape)
print(dataframe.describe())
len(dataframe)  # length of the dataframe (number of rows)

# changing the dataframe index
# dataframe = dataframe.set_index(dataframe['Name'])
dataframe.set_index(dataframe['Name'])
# selecting row by the Name
# print(dataframe.loc['Allen, Miss Elisabeth Walton'])

# loc is useful when the index of the DataFrame is a string
# iloc works by looking for the position in the dataframe regardless of whether the index is an integer or a string

# selecting row by condition (the first two women)
dataframe[dataframe['Sex'] == 'female'].head(2)
print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)].head(1))

# replacing values
dataframe['Sex'].replace("female", "Woman")
print(dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))
# replacing in the whole dataframe, not just one column
dataframe.replace(1, "One")
# replacing using regular expression
dataframe.replace(r"1st", "First", regex=True)

# renaming columns
dataframe = dataframe.rename(columns={'PClass': 'Passenger Class'})
# accepts dictionary; columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}
# looping through column names
for col in dataframe.columns:
    print(col)

# finding aggregated values
print('\nMaximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
# in the whole dataframe
print(dataframe.count())

# unique values
dataframe['Sex'].unique()               # list of unique values
dataframe['Sex'].value_counts()         # distinct values and their counts
dataframe['Passenger Class'].nunique()  # number of unique values

# finding missing values
print(dataframe[dataframe['Age'].isnull()].head(2))

# deleting a column
dataframe.drop('Age', axis=1)  # multiple columns: ['Age', 'Sex']
dataframe.drop(dataframe.columns[1], axis=1)  # column can be dropped by zero-indexed number

# deleting rows
dataframe[dataframe['Sex'] != 'male']  # here we are creating a new dataframe, excluding males
dataframe.drop([0, 1], axis=0)         # deleting first two rows
dataframe[dataframe.index != 0]        # deleting the first row (by row index)

# dropping duplicates
dataframe.drop_duplicates()  # considers a record duplicated if ALL columns match
dataframe.drop_duplicates(subset=['Sex'])  # this time only the Sex column has to match; keeps first record by default
dataframe.drop_duplicates(subset=['Sex'], keep='last')  # keeps last record

dataframe.duplicated(subset=['Sex'])  # shows if record is duplicated or not

# grouping by records
dataframe.groupby('Sex').mean()  # grouping rows by the values of column 'Sex' and calculating mean of each group
dataframe.groupby('Survived')['Name'].count()  # counting how many people survived and died
dataframe.groupby(['Sex', 'Survived'])['Age'].mean()  # grouping by two columns, then having an aggregated function

# looping through columns
for name in dataframe['Name'][0:2]:  # only first two columns
    print(name.upper())

print([name.upper() for name in dataframe['Name'][0:2]])

# applying function over elements in a column

# creating a function to return the uppercase of a string
def uppercase(x):
    return x.upper()

# Apply function, show two rows
print(dataframe['Name'].apply(uppercase)[0:2])

# applying a function to a group
print(dataframe.groupby('Sex').apply(lambda x: x.count()))


# inplace=True: edits the dataframe directly !!!!!


print('\n*****************************************************************************\n')


# grouping rows by time

time_index = pandas.date_range('06/06/2017', periods=100000, freq='30S')
# Create DataFrame
df_date = pandas.DataFrame(index=time_index)
# Create column of random values
df_date['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
df_date.resample('W').sum()  # grouping data weekly; 2W --> two weeks
print(df_date.resample('M').count())  # monthly


print('\n*****************************************************************************\n')


# creating a dataframe
df = pandas.DataFrame()
# adding columns
df['Name'] = ['Jacky Jackson', 'Steven Stevenson']
df['Age'] = [38, 25]
df['Driver'] = [True, False]

# creating another dataframe
df2 = pandas.DataFrame({'Name': ['Darth Vader'],
                        'Age': [55],
                        'Driver': [True]})

# creating another dataframe
df3 = pandas.DataFrame()
df3['Name'] = ['Anakin Skywalker']
df3['Age'] = [25]
df3['Driver'] = [True]

# appending df2 to df
df = df.append(df2, ignore_index=True)
# another way, concatenating df3 to df
df = pandas.concat([df, df3], axis=0)

# creating a new row
new_person = pandas.Series(['Molly Mooney', 40, False], index=['Name', 'Age', 'Driver'])
# appending the new row to the dataframe
df = df.append(new_person, ignore_index=True)
print(df)


print('\n*****************************************************************************\n')


# merging dataframes

# dataframe 1
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton']}
dataframe_employees = pandas.DataFrame(employee_data, columns=['employee_id', 'name'])
# dataframe 2
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pandas.DataFrame(sales_data, columns=['employee_id', 'total_sales'])

# merging the dataframes
print(pandas.merge(dataframe_employees, dataframe_sales, on='employee_id'))  # by default it is an INNER JOIN
print(pandas.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer'))  # with OUTER JOIN
print(pandas.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'))  # with LEFT JOIN (RIGHT)

# mergint by condition
pandas.merge(dataframe_employees, dataframe_sales, left_on='employee_id', right_on='employee_id')
