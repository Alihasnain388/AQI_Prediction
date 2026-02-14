import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import plotly.express as px

# 1. SETUP & AUTHENTICATION
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "Alihasnain388")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
mlflow.set_tracking_uri("https://dagshub.com/Alihasnain388/AQI_Model.mlflow")

# 2. PAGE CONFIGURATION
st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌤️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    .main { background-color: #f8fafc; }
    .aqi-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .daily-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. ASSET LOADING
@st.cache_resource(ttl=300)
def load_assets():
    client = mlflow.tracking.MlflowClient()
    MODEL_NAME = "Karachi_AQI_Model"
    versions = client.search_model_versions(filter_string=f"name='{MODEL_NAME}'")
    latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    scaler_path = mlflow.artifacts.download_artifacts(run_id=latest.run_id, artifact_path="model/scaler.pkl")
    scaler = joblib.load(scaler_path)
    return model, scaler, latest.version

@st.cache_data(ttl=300)
def get_latest_data():
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://ali321hasnain_db_user:etRWe1e6ASFlpwEO@cluster0.1eklm6h.mongodb.net/?appName=Cluster0")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["AQIPredictionSystem"]
    return db["karachi_features"].find_one(sort=[("_id", -1)])

# 4. MAIN DASHBOARD LOGIC
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🌤️ Karachi Real-Time AQI Dashboard</h1>", unsafe_allow_html=True)

try:
    model, scaler, model_ver = load_assets()
    data = get_latest_data()

    if data:
        # --- SECTION 1: CURRENT STATUS ---
        curr_aqi = data.get('aqi', 0)
        
        # Helper function for Status & Colors
        def get_aqi_status(val):
            if val > 250: return "#7e0023", "HAZARDOUS"
            elif val > 150: return "#ef4444", "UNHEALTHY"
            elif val > 100: return "#f97316", "SENSITIVE"
            elif val > 50: return "#eab308", "MODERATE"
            else: return "#22c55e", "GOOD"

        curr_color, curr_status = get_aqi_status(curr_aqi)

        st.markdown(f"""
            <div class="aqi-card" style="background-color:{curr_color};">
                <p style="font-size: 1.2rem; margin-bottom:0;">CURRENT AIR QUALITY INDEX</p>
                <h1 style="font-size: 5.5rem; color: white; margin:0; line-height:1;">{int(curr_aqi)}</h1>
                <p style="font-size: 1.8rem; font-weight: bold; margin:0;">{curr_status}</p>
                <p style="font-size: 0.8rem; opacity: 0.8; margin-top:10px;">Model Engine: v{model_ver}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- SECTION 2: 3-DAY DAILY FORECAST ---
        st.markdown("<h3 style='text-align: center;'>📅 3-Day Daily Outlook</h3>", unsafe_allow_html=True)
        
        # Prepare Prediction
        feature_cols = ["hour", "day_of_week", "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate", "Wind_Speed_kmh", "PM2.5_ugm3"]
        input_df = pd.DataFrame([data])[feature_cols]
        scaled_input = scaler.transform(input_df)
        predictions = model.predict(scaled_input)[0] 
        
        # Create Forecast DF
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        forecast_df = pd.DataFrame({
            "Time": [start_time + timedelta(hours=i) for i in range(72)],
            "AQI": predictions
        })
        
        # Daily Averages
        forecast_df['Day'] = forecast_df['Time'].dt.strftime('%A, %b %d')
        daily_avg = forecast_df.groupby('Day', sort=False)['AQI'].mean().reset_index()

        # Display Enhanced Daily Cards
        day_cols = st.columns(3)
        for i, row in daily_avg.head(3).iterrows():
            val = int(row['AQI'])
            d_color, d_status = get_aqi_status(val) # Reusing the helper function
            
            with day_cols[i]:
                st.markdown(f"""
                    <div class="daily-card" style="background-color:{d_color};">
                        <p style="margin:0; font-size: 1.1rem; font-weight: bold; opacity: 0.9;">{row['Day']}</p>
                        <h2 style="margin:5px 0; color: white; font-size: 2.5rem;">{val}</h2>
                        <p style="margin:0; font-weight: bold; font-size: 1rem;">{d_status}</p>
                        <p style="margin-top:10px; font-size: 0.7rem; opacity: 0.8;">DAILY AVERAGE</p>
                    </div>
                """, unsafe_allow_html=True)

        # --- SECTION 3: HOURLY TREND GRAPH ---
        st.markdown("<br><h3 style='text-align: center;'>📈 Hourly Forecast Trend</h3>", unsafe_allow_html=True)
        
        fig = px.area(forecast_df, x="Time", y="AQI", 
                      color_discrete_sequence=[curr_color],
                      labels={"AQI": "Predicted AQI", "Time": "Future Hour"})
        
        fig.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("No data found in MongoDB. Ensure your feature pipeline is running.")

except Exception as e:
    st.error(f"System Error: {e}")
