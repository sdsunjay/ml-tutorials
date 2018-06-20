import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016,12,31)

# df = web.DataReader('TSLA', 'morningstar', start, end)

# df.to_csv('tsla.csv')
filename = 'tsla.csv'
df = pd.read_csv(filename, parse_dates = True, index_col = "Date", na_values = ['nan', '0'])
print(df[['Open', 'High']].head())
# print(df.head())
# df['Close'].plot()

# plt.show()
