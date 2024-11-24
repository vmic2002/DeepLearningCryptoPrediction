import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
#TODO play around with hyperparameters like hidden_size for example, and seq_len maybe although 3 seems to work well
#TODO take into account MARKET SENTIMENT (POSITIVE VS NEGATIVE VIEW ONLINE ABOUT PRICE) + ANYTHING ELSE THAT COULD INFLUENCE BITCOIN PRICE, LOOK UP ALL FACTORS FOR BITCOIN PRICE, MACROECONOMIC INDICATORS

#PARAMS:
seq_len = 2 # (3 works very well) look at the past seq_len days to predict next value
features_for_prediction=['Close', 'Open', 'High', 'Low', 'Volume']  # Multiple features
# WE ARE TRYING TO PREDICT ALL FEATURES in features_for_prediction

csv = 'coin_Bitcoin.csv'
# Model parameters
Loss_type = 1 #1: MSE, 2:L1, 3:CrossEntropy 
LSTM_type = 1 #1: LSTM, 2: CNNLSTM, 3: BidirectionalLSTM also work
input_size = len(features_for_prediction)  # Subset of features (for example [Close, Volume, High, Low, Open])
hidden_size = 50  # Number of LSTM units
num_layers = 2
lr = 0.01 # learning rate
num_epochs = 80


class Loss:
    @staticmethod
    def get(type):
        if type == 1:
            return nn.MSELoss()
        elif type == 2:
            return nn.L1Loss()
        elif type == 3:
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid loss type. Choose 'MSE', 'MAE', or 'CrossEntropy'.")

class LSTM:
    @staticmethod
    def init(type):
        if type == 1:
            return LSTMModel()
        elif type == 2:
            return CNN_LSTMModel()
        elif type == 3:
            return BidirectionalLSTMModel()
        else:
            raise ValueError("Invalid LSTM type. Choose 'LSTM', 'CNNLSTM', or 'BidirectionalLSTM'.")

class CNN_LSTMModel(nn.Module):
    def __init__(self, num_filters=64, kernel_size=3):
        super(CNN_LSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size of the CNN output
        cnn_output_size = (seq_len // 2) * num_filters
        
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # Changed to predict all features
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_size, seq_length) for CNN
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_filters) for LSTM
        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the output from the last time step
        return out

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(BidirectionalLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        # Note: The output size of the LSTM is now 2*hidden_size due to bidirectionality
        self.fc = nn.Linear(hidden_size * 2, input_size)
    
    def forward(self, x):
        # Passing through Bidirectional LSTM layer
        out, _ = self.lstm(x)
        # We take the output from the last time step from both directions
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, input_size) #predicts all features
        
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
    # 3 past data points, and the target will be the next data point.
    X_data, y_data = [], []
    for i in range(len(data) - seq_len):
        X_data.append(data[i:i+seq_len])  # Sequence of previous features
        y_data.append(data[i+seq_len]) # next values for all features
    return np.array(X_data), np.array(y_data)

# Load the CSV
df = pd.read_csv('data/'+csv)
df.dropna(inplace=True) # remove rows with missing data

# Remove the time portion from the 'Date' column
df['Date'] = pd.to_datetime(df['Date']).dt.date

# Set 'Date' as the index for time-series analysis
df.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = normalize_data(df, features_for_prediction, scaler)

#print(scaled_data.shape)

X_data, y_data = create_sequential_data(scaled_data, seq_len)
#print(X_data.shape)
#print(y_data.shape)

# need to manually split data into train and test because we dont want data to be shuffled
# for sequential data order is key, we want training data to be the first part of the data and test to be the second part
percent_training_data = 0.9
train_size = int(len(X_data) * percent_training_data)

# Split X_data and y_data into training and testing sets
X_train = X_data[:train_size]
X_test = X_data[train_size:]
y_train = y_data[:train_size]
y_test = y_data[train_size:]

# Create the model
model = LSTM.init(LSTM_type)

# Loss and optimizer
criterion = Loss.get(Loss_type)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# Training loop

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
predicted_target = scaler.inverse_transform(predictions.numpy())

# Inverse transform for the actual values
actual_target = scaler.inverse_transform(y_test)

# Now, you should have the correct inverse transformed values for predicted and actual target

# Plot actual vs predicted for each feature
plt.figure(figsize=(15, 10))
plt.suptitle('Predicted vs Actual Values for Bitcoin Features (on test data)', fontsize=20, fontweight='bold')
for i in range(input_size):
    feature = features_for_prediction[i]
    # Calculate MSE
    mse = mean_squared_error(actual_target[:, i], predicted_target[:, i])
    print(f"Mean Squared Error (MSE) for {feature}: {mse}")

    plt.subplot(3, 2, i + 1)
    plt.plot(df.index[-len(y_test):], actual_target[:, i], label='Actual ' + feature, color='blue')
    plt.plot(df.index[-len(y_test):], predicted_target[:, i], label='Predicted ' + feature, color='orange')
    plt.title(f'{feature} - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.legend()
    
plt.tight_layout()
plt.show()

###############################################
# Once we notice the model can generalize well enough to the test data, we should
# try to predict future values by doing recursive forecasting or iterative prediction
# for this we shoul train the model on the entire data
X_train_tensor = torch.tensor(X_data, dtype=torch.float32)
y_train_tensor = torch.tensor(y_data, dtype=torch.float32)

# Model parameters
model = LSTM.init(LSTM_type)
criterion = Loss.get(Loss_type)
optimizer = optim.Adam(model.parameters(), lr=lr)

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
num_future_steps = 365  # Number of days you want to predict, the greater the more errors will accumulate
predictions = []

#print(X_data.shape)
input_seq = X_data[-1, :, :]  # Get the last sequence (e.g., last 3 time steps for seq_len = 3)
# Shape: (seq_len, num_features)

input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    for _ in range(num_future_steps):
        # Get the prediction for the next value
        prediction = model(input_seq)
        predictions.append(prediction.numpy().flatten())     
        
        # Create new input sequence by removing the oldest time step and adding the prediction as the new one
        
        input_seq = torch.cat([input_seq[:, 1:, :], prediction.unsqueeze(1)], dim=1)

predictions = np.array(predictions)
predictions = scaler.inverse_transform(predictions)

# Plot the results
# we plot the actual values followed by the predictions

# Create future dates for predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_steps)

# Plot results
plt.figure(figsize=(15,10))
plt.suptitle('Cryptocurrency Prediction using Recursive Forecasting', fontsize=20)

for i in range(input_size):
    feature = features_for_prediction[i]
    plt.subplot(3, 2, i + 1)
    plt.plot(df.index[-len(y_data):], scaler.inverse_transform(y_data)[:, i], label='Actual ' + feature, color='blue')
    plt.plot(future_dates, predictions[:, i], label='Predicted ' + feature, color='orange')
    plt.title(f'{feature} - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.legend()

plt.show()
