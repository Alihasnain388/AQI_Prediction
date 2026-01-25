import os
import mlflow
import mlflow.sklearn
import joblib

# -------------------------------
# 1. Force credentials
# -------------------------------
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

# -------------------------------
# 2. HARD-SET tracking URI
# -------------------------------
mlflow.set_tracking_uri(
    "https://dagshub.com/Alihasnain388/AQI_Model.mlflow"
)

mlflow.set_experiment("Karachi_AQI_CICD")

# -------------------------------
# 3. Load artifacts
# -------------------------------
model = joblib.load("Random_Forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# 4. Log model
# -------------------------------
with mlflow.start_run(run_name="CI_CD_Auto_Train"):

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    mlflow.log_artifact("scaler.pkl", artifact_path="model")

    mlflow.log_param("pipeline", "ci_cd")
    mlflow.log_param("trigger", "scheduled")

print("✅ Model registered WITHOUT OAuth")

