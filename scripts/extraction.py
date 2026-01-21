import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. Configuration for Karachi
LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

# 2. Define Date Range (Last 60 days)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
current_hour_str = datetime.now().strftime('%Y-%m-%dT%H:00')

def get_karachi_data():
    print(f"Fetching data from {start_date} to {end_date}...")

    # --- A. Fetch Weather Data (Bulk) ---
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
        f"&timezone={TIMEZONE}"
    )
    w_res = requests.get(weather_url).json()
    w_data = w_res['hourly']

    # --- B. Fetch Air Quality Data (Bulk) ---
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5,pm10,nitrogen_dioxide,carbon_monoxide,sulphur_dioxide,ozone,us_aqi"
        f"&timezone={TIMEZONE}"
    )
    aq_res = requests.get(aq_url).json()
    aq_data = aq_res['hourly']

    # --- C. Combine into a Single DataFrame ---
    df = pd.DataFrame({
        'Date_Time': w_data['time'],
        'Temp_C': w_data['temperature_2m'],
        'Humidity_%': w_data['relative_humidity_2m'],
        'Wind_Speed_kmh': w_data['wind_speed_10m'],
        'Pressure_hPa': w_data['surface_pressure'],
        'US_AQI': aq_data['us_aqi'],
        'PM2.5_ugm3': aq_data['pm2_5'],
        'PM10_ugm3': aq_data['pm10'],
        'NO2_ugm3': aq_data['nitrogen_dioxide'],
        'CO_ugm3': aq_data['carbon_monoxide'],
        'SO2_ugm3': aq_data['sulphur_dioxide'],
        'O3_ugm3': aq_data['ozone']
    })

    # --- D. Filter out Forecasted (Future) Data ---
    # The API often returns data until the end of the current day.
    # This line keeps only the rows that have already happened.
    df = df[df['Date_Time'] <= current_hour_str]

    # --- E. Save to CSV ---
    filename = 'karachi_actual_historical_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"Success! Saved {len(df)} rows of actual data to {filename}")
    print(f"Latest entry in CSV: {df['Date_Time'].iloc[-1]}")

if __name__ == "__main__":
    get_karachi_data()