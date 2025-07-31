import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load the dataset
file_path = "C:\\Users\\THANUSH\\Desktop\\Air_Quality_Algo\\vishakapatnam_hourly_filled.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# Define the pollutants to predict
pollutants = ['no2_value', 'pm10_value', 'pm25_value', 'so2_value']

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[pollutants])

# Define sequence length (look-back window)
sequence_length = 24 * 7  # Use 7 days of hourly data to predict

# Create sequences for training
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
# We'll use the last 3 days of the available data for testing,
# and the rest for training.
# Given the prediction is for July 4th, 5th, 6th, we'll train on data up to July 3rd.
# The provided data ends on July 3rd, 2025 23:00:00+05:30.
# So, we'll use all available data for training and then predict.

# For demonstration, let's assume the last few days are for validation if needed,
# but for predicting beyond the dataset, we train on all.
# Let's take the last 7 days for a small validation set if we were to evaluate performance
# on unseen data within the provided range.
# For this specific request, we'll train on all data up to July 3rd.

# Model Architecture (LSTM)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(len(pollutants))
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
# Using a small number of epochs for demonstration. Increase for better performance.
model.fit(X, y, epochs=50, batch_size=32, verbose=0)

# Prepare input for prediction (last 'sequence_length' data points from the dataset)
last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, len(pollutants))

# Predict for the next three days (72 hours)
future_predictions_scaled = []
current_input_sequence = last_sequence

for _ in range(24 * 3):  # Predict for 3 days * 24 hours
    next_hour_prediction_scaled = model.predict(current_input_sequence, verbose=0)
    future_predictions_scaled.append(next_hour_prediction_scaled[0])

    # Update the input sequence by removing the oldest data point and adding the new prediction
    current_input_sequence = np.append(current_input_sequence[:, 1:, :], next_hour_prediction_scaled.reshape(1, 1, len(pollutants)), axis=1)

# Inverse transform the predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled))

# Generate future timestamps
last_timestamp = df.index[-1]
future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(24 * 3)]

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(future_predictions, columns=pollutants, index=future_timestamps)

# Filter predictions for July 4th, 5th, and 6th
july_4th_start = datetime(2025, 7, 4, 0, 0, 0, tzinfo=df.index.tz)
july_6th_end = datetime(2025, 7, 6, 23, 0, 0, tzinfo=df.index.tz)

target_predictions = predictions_df[(predictions_df.index >= july_4th_start) & (predictions_df.index <= july_6th_end)]
print(target_predictions)
