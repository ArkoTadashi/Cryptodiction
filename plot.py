import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('apy.csv', parse_dates=['Date'])

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Actual'], label='Actual Price', color='blue', linewidth=2)
plt.plot(data['Date'], data['Predicted'], label='Predicted Price', color='orange', linestyle='--', linewidth=2)

plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('actual_vs_predicted_prices.png', dpi=300)
plt.show()
