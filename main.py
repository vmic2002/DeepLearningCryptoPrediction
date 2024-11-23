import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
#TODO, need to make LSTMModel more sohpisticated and play around with hyperparameters like hidden_size for example, and seq_len maybe although 3 seems to work well
#TODO take into account MARKET SENTIMENT (POSITIVE VS NEGATIVE VIEW ONLINE ABOUT PRICE) + ANYTHING ELSE THAT COULD INFLUENCE BITCOIN PRICE, LOOK UP ALL FACTORS FOR BITCOIN PRICE, MACROECONOMIC INDICATORS
#PARAMS:
seq_len = 4 # (3 works very well) look at the past seq_len days to predict next value
features_for_prediction=['Close', 'Open', 'High', 'Low', 'Volume']  # Multiple features
target = 'Close' #value we are predicting, has to be part of features_for_prediction
target_index = features_for_prediction.index(target)
csv = 'coin_Bitcoin.csv'
# Model parameters
input_size = len(features_for_prediction)  # Subset of features (for example [Close, Volume, High, Low, Open])
output_size = 1  # Predicting one value (next closing price for example if target = 'Close')
hidden_size = 200  # Number of LSTM units
num_layers = 2
lr = 0.01 # learning rate
num_epochs = 80

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    #TODO NEED TO SOPHISTICATE
    def forward(self, x):
        # Passing through LSTM layer
        out, _ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        out = self.fc(out)
        return out


def normalize_data(df, features, scaler):
    # features are subset of [Close, Open, High, Low, Volume], columns in df
    data = df[features].values
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def create_sequential_data(data, seq_len):
    # example: for a sequence length of 3, each input to the model will consist of
    # 3 past closing prices, and the target will be the next closing price.
    X_data, y_data = [], []
    for i in range(len(data) - seq_len):
        X_data.append(data[i:i+seq_len])  # Sequence of previous features
        y_data.append(data[i+seq_len, target_index])
    return np.array(X_data), np.array(y_data)






# Load the CSV
df = pd.read_csv('data/'+csv)
df.dropna(inplace=True) # remove rows with missing data

# Remove the time portion from the 'Date' column
df['Date'] = pd.to_datetime(df['Date']).dt.date


# Set 'Date' as the index for time-series analysis
df.set_index('Date', inplace=True)

# View the first few rows
#print(df.head())
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = normalize_data(df, features_for_prediction, scaler)

print(scaled_data.shape)

print("Creating sequential data...")
X_data, y_data = create_sequential_data(scaled_data, seq_len)


print(X_data.shape)


# Reshape x_data to be in the format (batch_size, seq_len, num_features)
# LSTM expects a 3D tensor as input: (batch_size, sequence_length, num_features).
#X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], input_size))

#print(X_data.shape)
#exit(1)

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
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=lr)


# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
print("Training LSTM model...")
# Training loop

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train_tensor)
    # Modify the loss calculation to match the target shape
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))  # Add the second dimension to match output shape

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
# Prepare arrays with zeros for the other features
predicted_target = scaler.inverse_transform(np.hstack((predictions.numpy(), np.zeros((predictions.shape[0], input_size - 1)))))

# Inverse transform for the actual values
actual_target = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], input_size - 1)))))

# Now, you should have the correct inverse transformed values for predicted and actual target


# Calculate MSE
mse = mean_squared_error(actual_target[:, target_index], predicted_target[:, target_index])

print(f"Mean Squared Error (MSE): {mse}")




# Plot the actual vs predicted prices
plt.figure(figsize=(12, 8))
plt.plot(actual_target[:, target_index], label='Actual '+target)
plt.plot(predicted_target[:, target_index], label='Predicted '+target)
plt.title('LSTM Model - Actual vs Predicted '+target+' of Test Data to see how well model generalizes')
plt.xlabel('Time')
plt.ylabel(target)
plt.legend()
plt.show()

#exit(1)

###############################################
# Once we notice the model can generalize well enough to the test data, we should
# try to predict future values by doing recursive forecasting or iterative prediction
# for this we shoul train the model on the entire data
X_train_tensor = torch.tensor(X_data, dtype=torch.float32)
y_train_tensor = torch.tensor(y_data, dtype=torch.float32)

# Model parameters
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# now model is trained on entire dataset, we can start the recursive forecasting!

# Predict future values
model.eval()
num_future_steps = 365  # Number of days you want to predict, the greater the more errors will accumulate
predictions = []
#print("Xdata shape:")
#print(X_data.shape)

input_seq = X_data[-1, :, :]  # Get the last sequence (e.g., last 3 time steps for seq_len = 3)
# Shape: (seq_len, num_features)


print(input_seq.shape)
input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
print(input_seq.shape)


with torch.no_grad():
    for _ in range(num_future_steps):
        # Get the prediction for the next value
        prediction = model(input_seq)
        #print(prediction.shape)        
        predictions.append(prediction.item())     
        
        # Get the last time step from the input sequence
        last_known_values = input_seq[0, -1, :].clone()
        
        # Update only the target
        last_known_values[target_index] = prediction.item()
       
        # since we are only updating the target value and reusing all other values, the predictions will quickly converge and flatten. TODO UPDATE MODEL TO PREDICT ALL VALUES (ALL FEATURES FOR PREDICTION)

        #print(last_known_values.shape)
        #print(input_seq[:,1:,:].shape)
        #exit(1)
         
        # Create new input sequence by removing the oldest time step and adding the new one
        input_seq = torch.cat([input_seq[:, 1:, :], last_known_values.unsqueeze(0).unsqueeze(0)], dim=1)     
    

# Inverse transform predictions if necessary
#predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

dummy = np.zeros((len(predictions), input_size))
dummy[:, target_index] = predictions
#predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
predictions = scaler.inverse_transform(dummy)[:, target_index]

# Plot the results
# we plot the actual values followed by the predictions
actual_target_color='blue'
predicted_target_color='orange'

# Get the true values
true_values = df[target].values

# Create future dates for predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_steps)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(df.index, true_values, label='True '+target, color=actual_target_color)
plt.plot(future_dates, predictions, label='Predicted '+target, color=predicted_target_color)
plt.title('Historical Data and Recursive Forecasting')
plt.xlabel('Date')
plt.ylabel(target)
plt.legend()
plt.show()
