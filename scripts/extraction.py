import requests
import datetime
import time
import csv

# --- CONFIGURATION ---
API_KEY = "312b4836ceff4d1d77eaec2597a2d8a7"
LAT = 24.8607  # Karachi
LON = 67.0011

# Calculate Unix timestamps for the last 90 days
end_time = int(time.time())
start_time = end_time - (90 * 24 * 60 * 60)

url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"

def save_pollution_data():
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        filename = "karachi_pollution_hourly.csv"
        records = data.get('list', [])
        
        if not records:
            print("No data found.")
            return

        # Column Headers
        headers = ['Date_Time', 'AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

            for entry in records:
                # Format: Year-Month-Day Hour:Minute (e.g., 2026-01-11 13:00)
                # No raw Unix timestamp included
                clean_date_time = datetime.datetime.fromtimestamp(entry['dt']).strftime('%Y-%m-%d %H:%M')
                
                comp = entry['components']
                
                writer.writerow([
                    clean_date_time, 
                    entry['main']['aqi'],
                    comp.get('co'), 
                    comp.get('no'), 
                    comp.get('no2'),
                    comp.get('o3'), 
                    comp.get('so2'), 
                    comp.get('pm2_5'),
                    comp.get('pm10'), 
                    comp.get('nh3')
                ])

        print(f"Success! Saved {len(records)} hourly rows to {filename}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    save_pollution_data()
