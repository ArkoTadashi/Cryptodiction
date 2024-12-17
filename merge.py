import pandas as pd

data1 = pd.read_csv('final_data.csv', parse_dates=['Date'])
data2 = pd.read_csv('WithoutScaling.csv', parse_dates=['Date'])

merged_data = pd.merge(data1, data2, on='Date', how='inner')

merged_data = merged_data[['Date', 'Actual', 'Predicted', 'RSI']]

merged_data.to_csv('apy.csv', index=False)

