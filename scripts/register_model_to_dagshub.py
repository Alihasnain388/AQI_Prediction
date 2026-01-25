import os

# -------------------------------------------------
# 1. Read credentials from ENV (CI/CD SAFE)
# -------------------------------------------------
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

import dagshub
import mlflow
import mlflow.sklearn
import joblib

# -------------------------------------------------
# 2. Connect to DagsHub MLflow
# -------------------------------------------------
dagshub.init(
    repo_owner="Alihasnain388",
    repo_name="AQI_Model",
    mlflow=True
)

# -------------------------------------------------
# 3. Load trained artifacts
# -------------------------------------------------
model_path = "Random_Forest_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ {model_path} not found")

model = joblib.load(model_path)

# -------------------------------------------------
# 4. Log & Register Model + Scaler
# -------------------------------------------------
with mlflow.start_run(run_name="Random_Forest_with_Scaler"):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    if os.path.exists(scaler_path):
        mlflow.log_artifact(scaler_path, artifact_path="model")

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("training_source", "MongoDB latest features")

print("🚀 Model & Scaler safely registered on DagsHub")
