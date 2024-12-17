import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from performer_pytorch import Performer
from statsmodels.tsa.stattools import acf
from sklearn.feature_selection import mutual_info_regression

data = pd.read_csv('WithoutScaling.csv', parse_dates=['Date'])
data.sort_values(by='Date', inplace=True)

features = ['Volume', 'RSI', 'EMA', 'MACD', 'MACD_Signal', 'ATR', 'BollingerMiddle',
            'BollingerUpper', 'BollingerLower', 'STD', 'UpCandle', 'DownCandle',
            'Dominance', 'DXY', 'FearGreed', 'HashRate']
label = 'Price'

scaler_features = StandardScaler()
scaler_label = StandardScaler()
data[features] = scaler_features.fit_transform(data[features])
data[[label]] = scaler_label.fit_transform(data[[label]])

lag_acf = acf(data[label], nlags=40)
optimal_lag = np.argmax(lag_acf < 0.2)

for lag in range(1, optimal_lag):
    for feature in features + [label]:
        data[f'{feature}_lag{lag}'] = data[feature].shift(lag)

for feature in features:
    data[f'{feature}_rolling_mean'] = data[feature].rolling(window=7).mean()
    data[f'{feature}_rolling_std'] = data[feature].rolling(window=7).std()

for feature in features:
    data[f'{feature}_rolling_mean_30'] = data[feature].rolling(window=30).mean()
    data[f'{feature}_rolling_std_30'] = data[feature].rolling(window=30).std()

data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month

data.dropna(inplace=True)

mi_scores = mutual_info_regression(data[features], data[label])
important_features = [features[i] for i in np.argsort(mi_scores)[-25:]]
data = data[important_features + [label, 'Date', 'day_of_week', 'month']]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len - 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

train_data = data[data['Date'] <= '2023-01-30']
test_data = data[(data['Date'] > '2023-01-30') & (data['Date'] <= '2024-10-25')]

train_X = train_data.drop(columns=['Date', 'Price']).values
train_y = train_data['Price'].values
test_X = test_data.drop(columns=['Date', 'Price']).values
test_y = test_data['Price'].values

seq_len = 90
train_dataset = TimeSeriesDataset(train_X, train_y, seq_len)
test_dataset = TimeSeriesDataset(test_X, test_y, seq_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class PerformerBiLSTMModel(nn.Module):
    def __init__(self, input_dim, embed_dim=256, lstm_hidden_dim=256, output_dim=1, dropout_prob=0.35):
        super(PerformerBiLSTMModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.performer = Performer(dim=embed_dim, depth=8, heads=8, dim_head=embed_dim // 8, causal=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.performer(x)
        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PerformerBiLSTMModel(train_X.shape[1])
model = model.to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

model.train()
best_loss = float('inf')
early_stopping_counter = 0
for epoch in range(100):
    epoch_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter > 7:
            print("Early stopping...")
            break

model.load_state_dict(best_model_state)

predictions, actuals = [], []
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze().item()
        scaled_output = scaler_label.inverse_transform([[outputs]])[0, 0]
        scaled_target = scaler_label.inverse_transform([[targets.item()]])[0, 0]
        predictions.append(scaled_output)
        actuals.append(scaled_target)

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

results = pd.DataFrame({'Date': test_data['Date'].iloc[seq_len - 1:].values, 'Actual': actuals, 'Predicted': predictions})
results.to_csv('final_data.csv', index=False)
