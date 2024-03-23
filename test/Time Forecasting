#pip install numpy pandas statsmodels matplotlib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(0)
time_index = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
data = pd.Series(100 + np.cumsum(np.random.randn(len(time_index))), index=time_index)

# Split data into training and testing sets
train_data = data['2022-01-01':'2022-10-31']
test_data = data['2022-11-01':]

# Statistical modeling with ARIMA
model = ARIMA(train_data, order=(1,1,1))  # Example order, adjust as needed
results = model.fit()
forecast_values = results.forecast(steps=len(test_data))

# Calculus-based modeling with linear regression
time_values = np.arange(len(train_data) + 1, len(train_data) + len(test_data) + 1).reshape(-1, 1)
X_train = np.arange(1, len(train_data) + 1).reshape(-1, 1)
y_train = train_data.values.reshape(-1, 1)
X_test = time_values.reshape(-1, 1)

# Fit linear regression model
regression_model = np.polyfit(X_train.flatten(), y_train.flatten(), 1)
forecast_values_regression = np.polyval(regression_model, X_test.flatten())

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Training Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(test_data.index, forecast_values, label='ARIMA Forecast', color='red', linestyle='--')
plt.plot(test_data.index, forecast_values_regression, label='Linear Regression Forecast', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Forecasting')
plt.legend()
plt.grid(True)
plt.show()
