import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
#TODO, need to make LSTMModel more sohpisticated and play around with hyperparameters like hidden_size for example, and seq_len maybe although 3 seems to work well
#TODO ask mathgpt if plot of predicted values looks correct according to code
#TODO take into account VOLUME, SENTIMENT ANALYSIS (POSITIVE VS NEGATIVE VIEW ONLINE ABOUT PRICE) + ANYTHING ELSE THAT COULD INFLUENCE BITCOIN PRICE, LOOK UP ALL FACTORS FOR BITCOIN PRICE
#PARAMS:
seq_len = 3 # (3 works very well) look at the past seq_len days to predict next value
column_to_predict='Close' #goes into func normalize_data
# Model parameters
input_size = 1  # One feature (for example Close, Volume, High, Low, Open...)
output_size = 1  # Predicting one value (next closing price)
hidden_size = 100  # Number of LSTM units


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    #TODO NEED TO SOPHISTICATE
    def forward(self, x):
        # Passing through LSTM layer
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out


def normalize_data(df, col_name, scaler):
    #col_name could be Close, Open, High, Low, Volume or any column in df
    # example: Extract the 'Close' column
    col = df[col_name].values
    col_reshaped = col.reshape(-1, 1)  # Reshape for the scaler

    # Normalize the data to the range [0, 1]
    scaled_prices = scaler.fit_transform(col_reshaped)
    return scaled_prices


def create_sequential_data(data, seq_len):
    # example: for a sequence length of 3, each input to the model will consist of
    # 3 past closing prices, and the target will be the next closing price.
    X_data, y_data = [], []
    for i in range(len(data) - seq_len):
        X_data.append(data[i:i+seq_len])  # Sequence of previous prices
        y_data.append(data[i+seq_len])  # Next price (target)
    return np.array(X_data), np.array(y_data)



# Load the CSV
df = pd.read_csv('data/coin_Bitcoin.csv')


# Remove the time portion from the 'Date' column
df['Date'] = pd.to_datetime(df['Date']).dt.date


# Set 'Date' as the index for time-series analysis
df.set_index('Date', inplace=True)

# View the first few rows
print(df.head())
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_prices = normalize_data(df, column_to_predict, scaler)

print("Creating sequential data...")
X_data, y_data = create_sequential_data(scaled_prices, seq_len)


# Reshape x_data to be in the format (batch_size, seq_len, num_features)
# LSTM expects a 3D tensor as input: (batch_size, sequence_length, num_features).
X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

print("Splitting data into train and test...")
# need to manually split data into train and test because we dont want data to be shuffled
# for sequential data order is key, we want training data to be the first part of the data and test to be the second part
percent_training_data = 0.9
train_size = int(len(X_data) * percent_training_data)
test_size = len(X_data) - train_size

# Split X_data and y_data into training and testing sets
X_train = X_data[:train_size]
X_test = X_data[train_size:]
y_train = y_data[:train_size]
y_test = y_data[train_size:]



print("Creating LSTM model...")
# Create the model
model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
print("Training LSTM model...")
# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

print("Model is trained! Can start predicting on test data...")

# Model is trained, so can predict
# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Switch to evaluation mode and make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# Inverse transform to get the actual predicted values
predicted_prices = scaler.inverse_transform(predictions.numpy())
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the actual vs predicted prices
plt.figure(figsize=(12, 8))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('LSTM Model - Actual vs Predicted Prices of Test Data to see how well model generalizes')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


############################
# Once we notice the model can generalize well enough to the test data, we should
# try to predict future values by doing recursive forecasting or iterative prediction
# for this we shoul train the model on the entire data
X_train_tensor = torch.tensor(X_data, dtype=torch.float32)
y_train_tensor = torch.tensor(y_data, dtype=torch.float32)

# Model parameters
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# now model is trained on entire dataset, we can start the recursive forecasting!
# Predict future values
model.eval()
future_steps = 365  # Number of days you want to predict, the greater the more errors will accumulate
predicted_future = []
# Start with the last seq_len values from the dataset
input_seq = scaled_prices[-seq_len:].reshape(1, seq_len, 1) # need to reshape for LSTM model
# Predict future values
with torch.no_grad():
    for _ in range(future_steps):
        prediction = model(torch.tensor(input_seq, dtype=torch.float32))  # Get next value prediction
        predicted_future.append(prediction.item())  # Append the predicted value
        input_seq = np.roll(input_seq, -1, axis=1)  # Shift the sequence by one step
        input_seq[0, -1, 0] = prediction.item()  # Add the prediction as the next input value

# Inverse transform to get actual predicted values
predicted_future_prices = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))

# Plot the results
# we plot the actual values followed by the predictions
actual_prices_color='blue'
predicted_prices_color='orange'
plt.figure(figsize=(12, 8))
plt.plot(df.index, scaler.inverse_transform(scaled_prices), label='Historical Prices', color=actual_prices_color)
future_dates = pd.date_range(df.index[-1], periods=future_steps+1, freq='D')[1:]
plt.plot(future_dates, predicted_future_prices, label='Predicted Future Prices', color=predicted_prices_color)#,linestyle='--')
plt.title('Bitcoin Price Prediction for the Next 2 Years')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


#####################################################################
#####################################################################
"""
#print("plotting close prices")
# Plot the Close prices
plt.figure(figsize=(10, 6))
df['Close'].plot(title='Actual Bitcoin Closing Prices Over Time', ylabel='Close Price (USD)', xlabel='Date')
plt.grid()
plt.show()

# Plot High, Low, and Close prices
df[['High', 'Low', 'Close']].plot(title='Bitcoin Price Trends', ylabel='Price (USD)', xlabel='Date', figsize=(12,8))
plt.grid()
plt.show()

# Plot volume and market cap
df[['Volume', 'Marketcap']].plot(title='Volume and market cap', ylabel='Price (USD)', xlabel='Date', figsize=(12,8))
plt.grid()
plt.show()



"""
