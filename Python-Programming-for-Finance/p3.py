import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
filename = 'tsla.csv'
df = pd.read_csv(filename, parse_dates = True, index_col = "Date", na_values = ['nan', '0'])
# print(df[['Open', 'High']].head())
# 100 moving average
df['100ma'] = df['Close'].rolling(window=100, min_periods=0).mean()
# we drop first 100 days
# df.dropna(inplace=True)
# we could also fillna
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Close'])
ax1.plot(df.index, df['100ma'])
ax2.plot(df.index, df['Volume'])

plt.show()
