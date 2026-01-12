import pandas as pd

# Load the raw 90-day hourly air pollution data for Karachi
# Ensure the file 'karachi_pollution_hourly.csv' is in your current directory
df = pd.read_csv('data/karachi_pollution_hourly.csv')

# 1. Convert Date_Time to a datetime object and sort to ensure time continuity
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df = df.sort_values('Date_Time')

# 2. FEATURE ENGINEERING

# Feature A: Hour of the Day (0-23)
# Captured because air pollution in Karachi follows a diurnal cycle (peaks at night/morning)
df['Hour'] = df['Date_Time'].dt.hour

# Feature B: Day of the Week (0=Monday, 6=Sunday)
# Captured to account for lower industrial/traffic emissions on weekends
df['Day_of_Week'] = df['Date_Time'].dt.dayofweek

# Feature C: AQI Last Hour (Lag 1)
# Captured because air quality is "persistent" (if it's bad now, it's usually bad in an hour)
df['AQI_last_hour'] = df['AQI'].shift(1)

# Feature D: AQI Last 24 Hours (Lag 24)
# Captured to catch the pattern of what the air was like at this exact time yesterday
df['AQI_last_24h'] = df['AQI'].shift(24)

# Feature E: Current AQI
# This remains in the dataset as your "Target" (Y) for training models

# 3. CLEANUP
# Shift operations create 'NaN' (empty) values for the first few rows (e.g., first 24 hours).
# We remove these to ensure the model only learns from complete data points.
final_df = df[['Date_Time', 'Hour', 'Day_of_Week', 'AQI_last_hour', 'AQI_last_24h', 'AQI']].dropna()

# 4. SAVE THE ENGINEERED DATASET
final_df.to_csv('data/features.csv', index=False)

print("Feature Engineering Complete!")
print(f"File saved as: karachi_prediction_ready.csv")
print(f"Total records ready for training: {len(final_df)}")
print("\nPreview of the engineered data:")
print(final_df.head())