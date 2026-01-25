import dagshub
import mlflow
import mlflow.sklearn
import joblib
import os

# -----------------------------
# 1. Init DagsHub
# -----------------------------
dagshub.init(
    repo_owner="Alihasnain388",
    repo_name="AQI_Model",
    mlflow=True
)

# -----------------------------
# 2. Load local model + scaler
# -----------------------------
model = joblib.load("Random_Forest_model.pkl")
scaler_path = "scaler.pkl"

# -----------------------------
# 3. Log EVERYTHING properly
# -----------------------------
with mlflow.start_run(run_name="AQI_Model_FIXED_UPLOAD"):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Karachi_AQI_Model"
    )

    # Log scaler INSIDE model folder
    mlflow.log_artifact(
        scaler_path,
        artifact_path="model"
    )

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("scaler", "StandardScaler")

print("✅ Model + scaler uploaded CORRECTLY")