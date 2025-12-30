import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load data
data = pd.read_csv("C:\\Users\\THANUSH\\Desktop\\Datasets\\vishakapatnam_hourly_filled.csv")

# Extract relevant columns
SO2 = pd.to_numeric(data["so2_value"], errors='coerce')
NO2 = pd.to_numeric(data["no2_value"], errors='coerce')
PM10 = pd.to_numeric(data["pm10_value"], errors='coerce')
time = pd.to_datetime(data["datetime"])

# Handle missing values
SO2 = SO2.interpolate()
NO2 = NO2.interpolate()
PM10 = PM10.interpolate()

# Time vector
N = len(time)
t = np.arange(N)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, SO2)
plt.title("SO2 Pollutant (Raw)")
plt.ylabel("Concentration (µg/m³)")

plt.subplot(3, 1, 2)
plt.plot(t, NO2)
plt.title("NO2 Pollutant (Raw)")
plt.ylabel("Concentration (µg/m³)")

plt.subplot(3, 1, 3)
plt.plot(t, PM10)
plt.title("PM10 (Raw)")
plt.xlabel("Time (days)")
plt.ylabel("Concentration (µg/m³)")
plt.tight_layout()
plt.show()