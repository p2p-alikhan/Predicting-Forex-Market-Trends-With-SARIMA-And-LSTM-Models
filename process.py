import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

sns.set(style="darkgrid")

print("Downloading last 2 years data of EUR/USD from yahoo finance")
data = yf.download('EURUSD=X', start='2022-07-01', end='2024-07-01')
data = data[['Close']]
print(data.head(15))
print(data.tail(15))
data_smooth = data.rolling(window=5).mean().dropna()
data.loc[:, 'Close'] = pd.to_numeric(data['Close'], errors='coerce')

print("Analyzing close price along with the 20EMA, 50EMA, 200EMA, and Bollinger Bands for volatility")
data['20EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
data['50EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
data['200EMA'] = data['Close'].ewm(span=200, adjust=False).mean()
data['MiddleBand'] = data['Close'].rolling(window=20).mean()
data['UpperBand'] = data['MiddleBand'] + 2 * data['Close'].rolling(window=20).std()
data['LowerBand'] = data['MiddleBand'] - 2 * data['Close'].rolling(window=20).std()
data['20_50_Crossover'] = ((data['20EMA'] > data['50EMA']) & (data['20EMA'].shift(1) <= data['50EMA'].shift(1))).astype(int)
data['50_200_Crossover'] = ((data['50EMA'] > data['200EMA']) & (data['50EMA'].shift(1) <= data['200EMA'].shift(1))).astype(int)
plt.figure(figsize=(14, 7))
sns.lineplot(data=data[['Close', '20EMA', '50EMA', '200EMA', 'MiddleBand', 'UpperBand', 'LowerBand']])
plt.title('Analyzing Volatility Using Bollinger Bands & Major Crossovers')
plt.legend(labels=['Close Price', '20EMA', '50EMA', '200EMA', 'Middle Band', 'Upper Band', 'Lower Band'])
plt.savefig('analyzing-volatility.png')

print("Calculating profitable setups using 20EMA, 50EMA, 200EMA crossovers")# 
plt.figure(figsize=(14, 7))
sns.lineplot(data=data[['Close', '20EMA', '50EMA', '200EMA']])
plt.scatter(data[data['20_50_Crossover'] == 1].index, 
            data['20EMA'][data['20_50_Crossover'] == 1], 
            marker='^', color='green', label='20/50 EMA Crossover')
plt.scatter(data[data['50_200_Crossover'] == 1].index, 
            data['50EMA'][data['50_200_Crossover'] == 1], 
            marker='^', color='red', label='50/200 EMA Crossover')
plt.title('Profitable Setups Using Major Crossovers')
plt.legend()
plt.savefig('profitable-setups.png')

print("Visualizing monthly returns for a quick view of performance across months and years")
data['Monthly Return'] = data['Close'].pct_change().resample('ME').sum()
monthly_return_matrix = data.pivot_table(values='Monthly Return', index=data.index.year, columns=data.index.month)
plt.figure(figsize=(14, 7))
sns.heatmap(monthly_return_matrix, annot=True, fmt=".2%", cmap='RdYlGn', center=0)
plt.title('EUR/USD Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Year')
plt.savefig('monthly-returns.png')

print("Analyzing the distribution of the close prices with a KDE plot for density estimation")
plt.figure(figsize=(14, 7))
sns.histplot(data['Close'], kde=True, bins=30)
plt.title('Distribution KDE Plot For Density Estimation')
plt.savefig('close-prices-distribution.png')

print("Calculating average close prices for each month for high level overview")
monthly_avg = data['Close'].resample('ME').mean()
plt.figure(figsize=(14, 7))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, color='blue')
plt.xticks(rotation=45)
plt.title('Monthly Average Close Prices Of EUR/USD')
plt.savefig('monthly-average-calculation.png')

print("Calculating autocorrelation & partial autocorrelation")
# ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) are tools used to measure and visualize the relationship between observations in a time series and their lagged values, with ACF showing correlations of current data with past data and PACF isolating the direct effect of past data on current data.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(data['Close'], lags=250, ax=axes[0])
plot_pacf(data['Close'], lags=250, ax=axes[1])
axes[0].set_title('ACF Plot')
axes[1].set_title('PACF Plot')
plt.savefig('ACF-PACF.png')

print("Splitting train & test into 80/20")
train_size = int(len(data_smooth) * 0.8)
train, test = data_smooth[:train_size], data_smooth[train_size:]

print("Using auto_arima to find best parameters for SARIMA model")

# This takes processing time and RAM, So i found heuristic below after lots of trials & errors
sarima_model = auto_arima(data_smooth,
                       start_p=1,
                       start_q=1,
                       max_p=3,
                       max_q=3,
                       m=15,
                       test='adf',
                       seasonal=True,
                       d=1,
                       D=1,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True,
                       n_fits=50,
                       method='lbfgs',
                       information_criterion='aic'
                       )                       

print("Using SARIMAX model")
sarima_fit = SARIMAX(train, order=sarima_model.order, seasonal_order=sarima_model.seasonal_order).fit()
sarima_pred = sarima_fit.get_forecast(steps=len(test)).predicted_mean # dynamic=False for more accuracy

sarima_mae = mean_absolute_error(test[:len(sarima_pred)], sarima_pred)
print('SARIMA Mean Absolute Error:', sarima_mae)

plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index[:len(sarima_pred)], sarima_pred, label='SARIMA Predictions')
plt.legend()
plt.title('SARIMA Predictions')
plt.savefig('sarima-predictions.png')

print("Forecasting using LSTM (Long short-term memory)")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data_smooth).reshape(-1, 1))

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def generateDatasetLstm(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 1
X_train, y_train = generateDatasetLstm(train_data, time_step)
X_test, y_test = generateDatasetLstm(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

lstm_pred = model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
lstm_mae = mean_absolute_error(test[-len(lstm_pred):], lstm_pred)
print('LSTM Mean Absolute Error:', lstm_mae)

plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train', linewidth=2.5)
plt.plot(test.index, test, label='Test', linewidth=2.5)
plt.plot(test.index[-len(lstm_pred):], lstm_pred, label='LSTM Predictions',  linewidth=2.5)
plt.legend()
plt.title('LSTM Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel('EUR/USD Exchange Rate', fontsize=14, fontweight='bold')
plt.savefig('lstm-predictions.png')
