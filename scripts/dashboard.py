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

# Custom CSS for UI styling (Hiding sidebar and styling cards)
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. ASSET LOADING FUNCTIONS
@st.cache_resource
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
    # Use your specific MONGO_URI secret here
    mongo_uri = os.getenv("MONGO_URI", "your_connection_string_here")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["AQIPredictionSystem"]
    # FIXED: Using _id sorting to guarantee latest data retrieval
    return db["karachi_features"].find_one(sort=[("_id", -1)])

# 4. MAIN DASHBOARD LOGIC
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🌤️ Karachi Real-Time AQI Monitor</h1>", unsafe_allow_html=True)

try:
    model, scaler, model_ver = load_assets()
    data = get_latest_data()

    if data:
        # --- SECTION 1: CURRENT STATUS ---
        curr_aqi = data.get('aqi_lag_1h', 0)
        
        # Determine Color Scale
        if curr_aqi > 150: color, status = "#ef4444", "UNHEALTHY"
        elif curr_aqi > 100: color, status = "#f97316", "SENSITIVE"
        elif curr_aqi > 50: color, status = "#eab308", "MODERATE"
        else: color, status = "#22c55e", "GOOD"

        st.markdown(f"""
            <div class="aqi-card" style="background-color:{color};">
                <p style="font-size: 1.2rem; margin-bottom:0;">CURRENT AIR QUALITY INDEX</p>
                <h1 style="font-size: 5rem; color: white; margin:0;">{int(curr_aqi)}</h1>
                <p style="font-size: 1.5rem; font-weight: bold;">{status}</p>
                <p style="font-size: 0.8rem; opacity: 0.8;">Model Engine: v{model_ver}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- SECTION 2: 3-DAY DAILY FORECAST ---
        st.markdown("### 📅 3-Day Daily Outlook")
        
        # Logic to predict 72 hours
        feature_cols = ["hour", "day_of_week", "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate", "Wind_Speed_kmh", "PM2.5_ugm3"]
        input_df = pd.DataFrame([data])[feature_cols]
        scaled_input = scaler.transform(input_df)
        predictions = model.predict(scaled_input)[0] 
        
        # Create Forecast DataFrame
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        forecast_df = pd.DataFrame({
            "Time": [start_time + timedelta(hours=i) for i in range(72)],
            "AQI": predictions
        })
        
        # Calculate Daily Averages
        forecast_df['Day'] = forecast_df['Time'].dt.strftime('%A, %b %d')
        daily_avg = forecast_df.groupby('Day', sort=False)['AQI'].mean().reset_index()

        # Display Daily Cards side-by-side
        day_cols = st.columns(3)
        for i, row in daily_avg.head(3).iterrows():
            with day_cols[i]:
                val = int(row['AQI'])
                # Small card color logic
                d_color = "#22c55e" if val < 50 else "#eab308" if val < 100 else "#ef4444"
                st.markdown(f"""
                    <div class="daily-card" style="background-color:{d_color};">
                        <p style="margin:0; font-weight: bold;">{row['Day']}</p>
                        <h2 style="margin:0; color: white;">{val}</h2>
                        <p style="margin:0; font-size: 0.7rem;">AVG PREDICTED AQI</p>
                    </div>
                """, unsafe_allow_html=True)

        # --- SECTION 3: HOURLY TREND GRAPH ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Hourly Forecast Trend (Next 72 Hours)")
        
        fig = px.area(forecast_df, x="Time", y="AQI", 
                      color_discrete_sequence=[color],
                      labels={"AQI": "Predicted AQI", "Time": "Future Hour"})
        
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Connection established, but no data found in MongoDB. Check your Feature Pipeline.")

except Exception as e:
    st.error(f"System Error: {e}")