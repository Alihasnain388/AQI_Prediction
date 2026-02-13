                                                                   Karachi AQI Prediction Dashboard

Firstly, Raw data like pollutants and weather were extracted from Open Meteo API hourly for last 2 months

Secondly, EDA was performed on the raw data to evaluate its relationship with AQI and to identify which features were impacting AQI the most, such as wind speed and PM2.5 in this case. These results help in feature engineering, which is one of the most important steps for achieving high model accuracy during training and prediction.

Based on the EDA results and the nature of the project, relevant features were engineered. Since this was a time-series forecasting problem, temporal features were particularly important, as AQI does not change abruptly under normal conditions. In most cases, the AQI of the previous hour is highly correlated with the AQI of the next hour, with only slight fluctuations. Significant variations typically occur only during unusual events such as fire outbreaks, which can cause sudden spikes.
The engineered features included: hour, day of the week, previous hour AQI, previous day AQI, AQI change rate, wind speed, PM2.5 concentration, and AQI summary.
Feature pipeline was than run against the last 2 month data and transport to Features Store. MongoDB was the feature store. I wanted hopswork but it was causing problems so i used MongoDB

Four models were selected for experimentation: Random Forest, Ridge Regression, Gradient Boosting, and a Neural Network (MLP). Feature scaling was applied using a standard scaler to normalize input variables, ensuring stable optimization and fair performance comparison across models, particularly for linear and neural network-based algorithms. Each model was implemented as a multi-output regressor to predict AQI for the next 72 hours (3 days) in a single run. Engineered features were retrieved from MongoDB and used for model training. The models were evaluated using R², MAE, and RMSE metrics. Random Forest was selected as the final model because it outperformed the others across all evaluation metrics, achieving an R² score of 0.87, an MAE of 5.79, and an RMSE of 9.07.

After training, SHAP analysis was performed to identify the most important features influencing the model’s predictions. The results showed that the previous hour’s AQI was the most significant feature. This was expected, as AQI prediction is a time-series problem and AQI values do not change abruptly under normal conditions. In most cases, the AQI of the next hour is closely related to the AQI of the previous hour, with only minor fluctuations, unless an unusual event occurs—such as a fire outbreak or a gas pipeline leak—which can cause a sudden and significant change in AQI.

Chosen model which was Random forest was than transported to Dagshub Model Registry. MLflow was used for experiment tracking, model versioning and registering the latest trained model in DagsHub.

The entire system was automated using CI/CD through GitHub Actions. The feature pipeline runs every hour to ensure that the feature store always contains the latest engineered features, while the training pipeline runs daily to ensure that the model registry always holds the most recently trained model based on the latest data.

For visualizing AQI, a Streamlit dashboard was developed that fetches the latest feature data from MongoDB and the latest trained model from the DagsHub model registry. The dashboard displays the current AQI, the average AQI for the next three days, and an hourly AQI prediction graph covering a three-day horizon. Different colors are used to represent AQI levels, such as green for good air quality and red for hazardous conditions.

Link: https://alihasnain388-aqi-prediction-scriptsdashboard-fep8bd.streamlit.app/


