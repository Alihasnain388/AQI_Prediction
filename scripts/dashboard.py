import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import plotly.express as px

# 1. SETUP & AUTHENTICATION (Mapping Secrets)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "Alihasnain388")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
mlflow.set_tracking_uri("https://dagshub.com/Alihasnain388/AQI_Model.mlflow")

# 2. PAGE CONFIGURATION
st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌤️", layout="wide")

# Custom Eye-Catchy CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #1E3A8A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; font-weight: 800; }
    .aqi-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA & MODEL ASSETS
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
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://ali321hasnain_db_user:etRWe1e6ASFlpwEO@cluster0.1eklm6h.mongodb.net/?appName=Cluster0")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["AQIPredictionSystem"]
    return db["karachi_features"].find_one(sort=[("_id", -1)])

# 4. DASHBOARD INTERFACE
st.title("🌤️ Karachi AQI Real-Time Monitor")

try:
    model, scaler, model_ver = load_assets()
    data = get_latest_data()

    if data:
        # --- CALULATE CURRENT AQI ---
        # Assuming your MongoDB field is 'aqi_lag_1h' or similar
        curr_aqi = data.get('aqi_lag_1h', 0)
        
        # Determine Color & Health Status (PDF Guideline Step 8)
        if curr_aqi > 250:
            color, status, msg = "#7e0023", "HAZARDOUS", "🚨 Severe Risk: Stay indoors and keep windows closed."
            st_func = st.error
        elif curr_aqi > 150:
            color, status, msg = "#ff0000", "UNHEALTHY", "⚠️ High Risk: Avoid outdoor activities."
            st_func = st.error
        elif curr_aqi > 100:
            color, status, msg = "#f97316", "UNHEALTHY FOR SENSITIVE GROUPS", "💡 Sensitive groups should limit outdoor time."
            st_func = st.warning
        elif curr_aqi > 50:
            color, status, msg = "#facc15", "MODERATE", "💡 Air quality is acceptable for most people."
            st_func = st.info
        else:
            color, status, msg = "#009966", "GOOD", "✅ Air quality is satisfactory. Enjoy the outdoors!"
            st_func = st.success

        # BIG EYE-CATCHY AQI CARD
        st.markdown(f"""
            <div class="aqi-card" style="background-color:{color};">
                <p style="font-size: 24px; margin-bottom: 0; opacity: 0.9;">CURRENT AIR QUALITY INDEX</p>
                <h1 style="color: white; font-size: 110px; margin: 0; line-height: 1;">{int(curr_aqi)}</h1>
                <p style="font-size: 32px; font-weight: bold; margin-top: 0;">{status}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Health Advice Alert
        st_func(msg)

        # --- 72-HOUR FORECAST SECTION ---
        st.markdown("## 📈 72-Hour AQI Trend Forecast")
        
        # Predict
        feature_cols = ["hour", "day_of_week", "aqi_lag_1h", "aqi_lag_24h", "aqi_change_rate", "Wind_Speed_kmh", "PM2.5_ugm3"]
        input_df = pd.DataFrame([data])[feature_cols]
        scaled_input = scaler.transform(input_df)
        predictions = model.predict(scaled_input)[0]
        
        # Time Axis
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        forecast_df = pd.DataFrame({
            "Time": [start_time + timedelta(hours=i) for i in range(72)],
            "AQI_Forecast": predictions
        })

        # Eye-Catchy Plotly Chart
        fig = px.area(forecast_df, x="Time", y="AQI_Forecast", 
                      title="Predicted Air Quality (Next 3 Days)",
                      color_discrete_sequence=[color]) 
        
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Future Time",
            yaxis_title="AQI Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Connection successful, but no feature data was found in MongoDB.")

except Exception as e:
    st.error(f"Unable to load the AQI system: {e}")

# Sidebar Info
st.sidebar.markdown(f"### 🚀 System Info")
st.sidebar.write(f"**Model Registry:** v{model_ver}")
st.sidebar.write(f"**City:** Karachi, PK")
st.sidebar.write("**Status:** 🟢 Live Tracking")