import pandas as pd
import quandl 

# Get google daily stock data
df = quandl.get('WIKI/GOOGL')

# Create dataframe with target data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
# Create dataframes with target metrics
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
#Create dataframe which contains targeted metrics
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print(df.head())