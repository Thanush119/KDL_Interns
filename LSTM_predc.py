import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_csv("C:\\Users\\THANUSH\\Desktop\\Datasets\\vishakapatnam_hourly_filled.csv", parse_dates=['datetime'])
df.set_index('datetime', inplace=True)
daily_df = df.resample('D').mean()
daily_df = daily_df.ffill()
pollutants = ['pm25_value', 'pm10_value', 'no2_value', 'so2_value']
for pollutant in pollutants:
    for lag in range(1, 8):
        daily_df[f'{pollutant}lag{lag}'] = daily_df[pollutant].shift(lag)
daily_df = daily_df.dropna()
train = daily_df.iloc[:-10]
test = daily_df.iloc[-10:]
models = {}
features = [f'{p}lag{l}' for p in pollutants for l in range(1, 8)]
for pollutant in pollutants:
    X_train = train[features]
    y_train = train[pollutant]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models[pollutant] = model
last_row = test.iloc[-1]
predictions = {p: [] for p in pollutants}
input_row = last_row.copy()
for _ in range(3):
    input_features = input_row[features].values.reshape(1, -1)
    for pollutant in pollutants:
        pred = models[pollutant].predict(input_features)[0]
        predictions[pollutant].append(pred)
        for lag in range(7, 1, -1):
            input_row[f'{pollutant}lag{lag}'] = input_row[f'{pollutant}lag{lag-1}']
        input_row[f'{pollutant}_lag_1'] = pred
for pollutant in pollutants:
    print(f"Next 3 days predicted {pollutant}: {predictions[pollutant]}")