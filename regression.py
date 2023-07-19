import math

import numpy as np
import pandas as pd
import quandl
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression

# Get google daily stock data
df = quandl.get('WIKI/GOOGL')

# Create dataframe with target data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

# Create dataframes with target metrics
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Create dataframe which contains targeted metrics
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# Column to predict future values
forecast_col = 'Adj. Close'
# To avoid losing data with holes, replace NaN holes with -9999 to force outlier, but retain rest of data.
df.fillna(-9999, inplace=True)

# Attempt to forecast 1% out from data
forecast_out = int(math.ceil(0.01*len(df)))

# Shift label column by -forecast percent, causing label column data in row to be that value into future
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())

X = np.array(df.drop(['label']))