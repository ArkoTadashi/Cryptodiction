import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('apy.csv', parse_dates=['Date'])

initial_balance_usdt = 1000.0
usdt_balance = initial_balance_usdt
btc_balance = 0.0
average_buy_price = 0.0

usdt_min_percentage = 0.2 
usdt_max_percentage = 0.9 
rsi_oversold = 30
rsi_overbought = 70
price_diff_threshold = 0.05  

portfolio_values = []

def calculate_portfolio_value(actual_price, btc_balance, usdt_balance):
    return btc_balance * actual_price + usdt_balance

def scale_transaction(base_amount, rsi, price_diff):
    rsi_scale = (70 - rsi) / 40 if rsi < 50 else (rsi - 30) / 40 
    diff_scale = abs(price_diff) * 10
    return base_amount * rsi_scale * diff_scale

def calculate_apy(start_balance, end_balance, days):
    return ((end_balance / start_balance) ** (365 / days) - 1) * 100

for i in range(1, len(data)):
    today_price = data.loc[i - 1, 'Actual'] 
    next_day_price_predicted = data.loc[i, 'Predicted'] 
    rsi_today = data.loc[i - 1, 'RSI']

    price_diff_percentage = (next_day_price_predicted - today_price) / today_price
    total_portfolio_value = calculate_portfolio_value(today_price, btc_balance, usdt_balance)
    min_usdt_reserve = total_portfolio_value * usdt_min_percentage  

    if rsi_today < rsi_oversold or price_diff_percentage > price_diff_threshold:
        available_usdt = usdt_balance - min_usdt_reserve
        if available_usdt > 0:
            buy_amount_usdt = scale_transaction(available_usdt, rsi_today, price_diff_percentage)
            buy_amount_usdt = min(buy_amount_usdt, available_usdt)
            if buy_amount_usdt > 0:
                btc_bought = buy_amount_usdt / today_price
                btc_balance += btc_bought
                usdt_balance -= buy_amount_usdt
                average_buy_price = ((average_buy_price * (btc_balance - btc_bought)) + (btc_bought * today_price)) / btc_balance
                print(f"Day {i}: BUY | Price: {today_price:.2f}, RSI: {rsi_today:.2f}, BTC Bought: {btc_bought:.6f}, USDT: {usdt_balance:.2f}")

    elif rsi_today > rsi_overbought or price_diff_percentage < -price_diff_threshold: 
        sell_amount_btc = scale_transaction(btc_balance, rsi_today, price_diff_percentage)
        sell_amount_btc = min(sell_amount_btc, btc_balance)

        potential_usdt_balance = usdt_balance + (sell_amount_btc * today_price)
        max_usdt_allowed = total_portfolio_value * usdt_max_percentage

        if potential_usdt_balance > max_usdt_allowed:
            sell_amount_usdt = max_usdt_allowed - usdt_balance
            sell_amount_btc = sell_amount_usdt / today_price 
            sell_amount_btc = min(sell_amount_btc, btc_balance)

        if sell_amount_btc > 0:
            sell_amount_usdt = sell_amount_btc * today_price
            btc_balance -= sell_amount_btc
            usdt_balance += sell_amount_usdt
            print(f"Day {i}: SELL | Price: {today_price:.2f}, RSI: {rsi_today:.2f}, BTC Sold: {sell_amount_btc:.6f}, USDT: {usdt_balance:.2f}")

    portfolio_value = calculate_portfolio_value(today_price, btc_balance, usdt_balance)
    portfolio_values.append({
        'Date': data.loc[i, 'Date'],
        'Portfolio_Value': portfolio_value,
        'BTC_Balance': btc_balance,
        'USDT_Balance': usdt_balance
    })

results_df = pd.DataFrame(portfolio_values)
results_df.to_csv('paper_trading_results.csv', index=False)

final_portfolio_value = calculate_portfolio_value(data.iloc[-1]['Actual'], btc_balance, usdt_balance)
trading_days = len(data) - 1
apy = calculate_apy(initial_balance_usdt, final_portfolio_value, trading_days)

print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Final USDT Balance: ${usdt_balance:.2f}")
print(f"Final BTC Balance: {btc_balance:.6f} BTC")
print(f"Average Buy Price: ${average_buy_price:.2f} per BTC")
print(f"Annual Percentage Yield (APY): {apy:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(results_df['Date'], results_df['Portfolio_Value'], label='Portfolio Value', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('portfolio_value_over_time.png', dpi=300)
plt.show()
