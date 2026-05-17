                                                                   Karachi AQI Prediction Dashboard

# 🌍 Karachi AQI Prediction System

An end-to-end machine learning system for predicting the **Air Quality Index (AQI) of Karachi for the next 72 hours (3 days)** using historical pollutant and weather data, automated ML pipelines, feature engineering, experiment tracking, and a real-time Streamlit dashboard.

Link: https://alihasnain388-aqi-prediction-scriptsdashboard-fep8bd.streamlit.app/

---

## 🚀 Project Overview

Air pollution has become a major environmental and public health issue in large urban cities such as Karachi. Monitoring and forecasting AQI can help individuals, healthcare organizations, and government authorities make proactive decisions.

This project focuses on building a complete production-oriented AQI forecasting pipeline using:

* Real-time pollutant & weather data
* Time-series feature engineering
* Multiple machine learning models
* Automated retraining pipelines
* Feature store integration
* MLflow experiment tracking
* DagsHub model registry
* Streamlit dashboard deployment
* GitHub Actions CI/CD automation

The system predicts AQI for the next **72 hours** in a single run using a multi-output regression approach.

---

# 🧠 Key Features

## 📡 Real-Time Data Collection

* Pollutant and weather data fetched hourly using the Open-Meteo API
* Historical Karachi AQI data collection
* Automated feature pipeline updates

## 📊 Exploratory Data Analysis (EDA)

* Correlation analysis between AQI and environmental factors
* AQI trend visualization
* Pollutant impact analysis
* Time-series behavior analysis

## ⚙️ Advanced Feature Engineering

Engineered features include:

* Hour of the day
* Day of the week
* Previous hour AQI
* Previous day AQI
* AQI change rate
* Wind speed
* PM2.5 concentration
* AQI summary features

## 🤖 Machine Learning Models

The following models were trained and evaluated:

* Random Forest Regressor
* Ridge Regression
* Gradient Boosting Regressor
* Neural Network (MLP Regressor)

## 📈 Model Evaluation

Models were evaluated using:

* R² Score
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

### 🏆 Best Performing Model

**Random Forest Regressor** achieved the best results:

| Metric   | Score |
| -------- | ----- |
| R² Score | 0.87  |
| MAE      | 5.79  |
| RMSE     | 9.07  |

## 🔍 SHAP Explainability

SHAP analysis was performed to understand feature importance.

### Most Important Feature

* Previous Hour AQI

This confirms the time-series nature of AQI forecasting where recent AQI values strongly influence future predictions.

---

# 🏗️ System Architecture

```text
                Open-Meteo API
                       │
                       ▼
            Data Collection Pipeline
                       │
                       ▼
               Feature Engineering
                       │
                       ▼
                 MongoDB Feature Store
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Random Forest   Gradient Boosting   MLP
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
                 MLflow Tracking
                       │
                       ▼
             DagsHub Model Registry
                       │
                       ▼
              Streamlit Dashboard
                       │
                       ▼
                 AQI Visualization
```

---

# 📂 Repository Structure

```bash
AQI_Prediction/
│
├── .github/workflows/              # CI/CD pipelines
├── artifacts/                      # Model artifacts
├── scripts/                        # Pipeline scripts
├── tmp/                            # Temporary files
│
├── EDA.ipynb                       # Exploratory Data Analysis
├── Prediction.ipynb                # Prediction workflow
├── Training.ipynb                  # Model training pipeline
├── Training_only_randomforest.ipynb
│
├── karachi_actual_historical_data.csv
├── karachi_final_features.csv
│
├── Random_Forest_model.pkl
├── Gradient_Boosting_model.pkl
├── Ridge_Regression_model.pkl
├── Neural_Network_MLP_model.pkl
├── scaler.pkl
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 🔬 Data Pipeline

## 1️⃣ Data Collection

Raw pollutant and weather data were extracted hourly from the Open-Meteo API.

Collected parameters include:

* PM2.5
* PM10
* Wind Speed
* Temperature
* Humidity
* AQI values

Historical data for the previous two months was used for training and forecasting.

---

## 2️⃣ Exploratory Data Analysis

EDA was conducted to:

* Identify highly correlated features
* Understand AQI trends over time
* Detect patterns and anomalies
* Evaluate seasonal and hourly AQI behavior

Key findings:

* PM2.5 had strong correlation with AQI
* Wind speed significantly impacted AQI movement
* Previous AQI values were highly predictive

---

## 3️⃣ Feature Engineering

Feature engineering was one of the most important stages of the project.

Since AQI forecasting is a time-series problem, temporal dependency features were introduced.

### Temporal Features

* Previous hour AQI
* Previous day AQI
* AQI rate of change
* Hour index
* Day-of-week patterns

### Environmental Features

* PM2.5 concentration
* Wind speed
* AQI category encoding

All engineered features were stored inside MongoDB acting as the Feature Store.

---

# 🧠 Model Training

## Multi-Output Regression

Instead of predicting a single AQI value, the models predict AQI values for the next:

* 72 hours
* 3 days ahead

This enables continuous future AQI forecasting.

## Feature Scaling

A Standard Scaler was used to normalize numerical features for:

* Stable optimization
* Improved neural network performance
* Fair comparison across models

---

# 📈 Model Evaluation

## Compared Models

| Model             | Purpose                       |
| ----------------- | ----------------------------- |
| Random Forest     | Ensemble-based regression     |
| Ridge Regression  | Linear regularized regression |
| Gradient Boosting | Sequential boosting algorithm |
| MLP Regressor     | Neural network forecasting    |

## Final Selected Model

Random Forest outperformed all models due to:

* Better generalization
* Lower prediction error
* Higher R² score
* Stability on time-series data

---

# 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) was used for model interpretability.

### Insights

* Previous AQI was the strongest predictor
* PM2.5 contributed significantly to AQI spikes
* Wind speed influenced pollutant dispersion

This analysis improved trust and explainability of predictions.

---

# ☁️ MLflow & DagsHub Integration

## MLflow

Used for:

* Experiment tracking
* Metric logging
* Model versioning
* Pipeline reproducibility

## DagsHub

Used as the centralized Model Registry.

The latest trained model is automatically pushed to DagsHub after successful training.

---

# ⚡ CI/CD Automation

GitHub Actions automates the entire system.

## Automated Pipelines

### ⏱️ Hourly Feature Pipeline

* Fetches latest data
* Performs feature engineering
* Updates MongoDB Feature Store

### 📅 Daily Training Pipeline

* Retrains models on latest data
* Logs experiments with MLflow
* Registers latest model to DagsHub

This ensures the system always stays updated with recent AQI trends.

---

# 📊 Streamlit Dashboard

A real-time Streamlit dashboard was developed for AQI visualization.

## Dashboard Features

* Current AQI display
* Next 3-day AQI forecast
* Hourly AQI trend graph
* AQI severity color indicators
* Real-time prediction visualization

### AQI Color Indicators

| AQI Level | Color    |
| --------- | -------- |
| Good      | Green    |
| Moderate  | Yellow   |
| Poor      | Orange   |
| Very Poor | Red      |
| Hazardous | Dark Red |

## 🌐 Live Dashboard

### Dashboard Link

   [https://alihasnain388-aqi-prediction-scriptsdashboard-fep8bd.streamlit.app/](https://alihasnain388-aqi-prediction-scriptsdashboard-fep8bd.streamlit.app/)

---

# 🛠️ Tech Stack

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* Random Forest
* Gradient Boosting
* Ridge Regression
* MLP Regressor
* SHAP

## Data Handling

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn
* Streamlit

## MLOps

* MLflow
* DagsHub
* GitHub Actions
* MongoDB

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/Alihasnain388/AQI_Prediction.git
cd AQI_Prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run Training Pipeline

```bash
python scripts/train.py
```

## Run Feature Pipeline

```bash
python scripts/feature_pipeline.py
```

## Run Dashboard

```bash
streamlit run scripts/dashboard.py
```

---

# 📊 Future Improvements

Potential future enhancements:

* LSTM and Transformer-based forecasting
* Real-time streaming data ingestion
* Multi-city AQI forecasting
* Docker containerization
* Kubernetes deployment
* Airflow orchestration
* Weather forecasting integration
* Mobile application deployment
* Alert notification system

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

* Time-series forecasting
* Machine learning model comparison
* Feature engineering
* Explainable AI
* MLOps workflows
* CI/CD automation
* Feature stores
* Model registries
* Streamlit deployment

---

# 🤝 Contribution

Contributions, improvements, and suggestions are welcome.

Feel free to fork the repository and submit pull requests.

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

## Syed Ali Hasnain Kazmi

* AI Engineer & ML Developer
* Focused on Machine Learning, MLOps, AI Systems, and Data Engineering

### GitHub

[Alihasnain388 GitHub](https://github.com/Alihasnain388?utm_source=chatgpt.com)

---

# ⭐ Support

If you found this project useful:

* Star the repository
* Fork the project
* Share it with others
* Contribute improvements

---

# 📌 Project Highlights

✅ End-to-End AQI Forecasting System
✅ Real-Time Data Pipeline
✅ Feature Store with MongoDB
✅ MLflow Experiment Tracking
✅ DagsHub Model Registry
✅ GitHub Actions Automation
✅ Streamlit Dashboard Deployment
✅ Explainable AI using SHAP
✅ Multi-Model Comparison
✅ 72-Hour AQI Forecasting

