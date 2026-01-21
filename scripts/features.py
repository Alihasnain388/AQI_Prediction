import pandas as pd

def create_features(file_path):
    # 1. Load your raw data
    df = pd.read_csv(file_path)
    
    # 2. Sort by time to ensure calculations (lags) are correct
    # (We need the datetime to sort, but we will drop it later)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df = df.sort_values('Date_Time')
    
    # 3. Time-based Features (As per PDF)
    df['hour'] = df['Date_Time'].dt.hour
    df['day_of_week'] = df['Date_Time'].dt.dayofweek
    
    # 4. Memory Features (Lags)
    # We use US_AQI from raw data to create these, then rename later
    df['aqi_lag_1h'] = df['US_AQI'].shift(1)
    df['aqi_lag_24h'] = df['US_AQI'].shift(24)
    
    # 5. Derived Feature (As per PDF)
    df['aqi_change_rate'] = df['US_AQI'].diff()
    
    # 6. Rename US_AQI to aqi as requested
    df = df.rename(columns={'US_AQI': 'aqi'})
    
    # 7. Select only the relevant features (Excluding Date_Time)
    selected_columns = [
        'hour', 
        'day_of_week', 
        'aqi_lag_1h', 
        'aqi_lag_24h', 
        'aqi_change_rate', 
        'Wind_Speed_kmh', 
        'PM2.5_ugm3', 
        'aqi'  # This is the renamed target
    ]
    
    feature_df = df[selected_columns]
    
    # 8. Clean up
    # Dropping rows where lags are empty (the first 24 hours)
    feature_df = feature_df.dropna()
    
    return feature_df

# Run the script
if __name__ == "__main__":
    data_path = 'karachi_actual_historical_data.csv'
    final_features = create_features(data_path)
    
    # Save the final cleaned version
    final_features.to_csv('karachi_final_features.csv', index=False)
    
    print("✅ Feature Table Modified Successfully!")
    print("Columns in file:", final_features.columns.tolist())
    print(final_features.head())