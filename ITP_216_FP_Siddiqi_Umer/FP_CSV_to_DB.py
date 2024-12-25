# WHAT THIS PYTHON FILE DOES
# Uses the csv file and converts it into a database.


import sqlite3 as sl

import pandas as pd


pd.set_option('display.max_columns', None)

# connecting to the 'nba' database
conn = sl.connect('nba.db')
curs = conn.cursor()

# Drops the table if it already exists and creates a new one with player, draft year, pick, and ppg columns
curs.execute('DROP TABLE IF EXISTS nba')
curs.execute('CREATE TABLE IF NOT EXISTS nba (`Player` text, `DraftYr` text, `Pk` number, `PPG` number )')
conn.commit()

# Read CSV file, and take the data from the columns of the csv files that you are going to use in the database
df = pd.read_csv('draft-data-20-years.csv', usecols=['Player', 'DraftYr', 'Pk', 'PPG'])
# Get rid of all null points per game values
df = df[df["PPG"].notnull()]

# Insert data into database
df.to_sql(name='nba', con=conn, if_exists='replace', index=False)

