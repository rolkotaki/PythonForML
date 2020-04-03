import pandas
from sqlalchemy import create_engine


# Create a connection to the database
database_connection = create_engine('mysql+pymysql://root:@localhost/turkeybase')

dataframe = pandas.read_sql_query('SELECT * FROM user', database_connection)

print(dataframe.head(1))  # first row
