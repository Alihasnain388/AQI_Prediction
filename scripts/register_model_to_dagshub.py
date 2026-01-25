import os
import mlflow
import mlflow.sklearn
import joblib

# -------------------------------------------------
# 1. Auth via ENV (NO BROWSER, CI/CD SAFE)
# -------------------------------------------------
DAGSHUB_USER = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not DAGSHUB_USER or not DAGSHUB_TOKEN:
    raise EnvironmentError("❌ DagsHub credentials not found in environment")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# -------------------------------------------------
# 2. Set MLflow Tracking URI (THIS IS THE KEY)
# -------------------------------------------------
mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/AQI_Model.mlflow"
)

mlflow.set_experiment("Karachi_AQI_Production")

# -------------------------------------------------
# 3. Load trained artifacts
# -------------------------------------------------
model = joblib.load("Random_Forest_model.pkl")
scaler_path = "scaler.pkl"

# -------------------------------------------------
# 4. Log & Register
# -------------------------------------------------
with mlflow.start_run(run_name="RF_AQI_Auto_Retrain"):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    if os.path.exists(scaler_path):
        mlflow.log_artifact(scaler_path, artifact_path="model")

    mlflow.log_param("training_type", "automated_ci_cd")
    mlflow.log_param("data_source", "latest_mongodb_features")

print("✅ CI/CD Model Registration Successful")
