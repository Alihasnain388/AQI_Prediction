                                                                   Karachi AQI Prediction Dashboard

Firstly, Raw data like pollutants and weather were extracted from Open Meteo API hourly for last 2 months

Secondly on this raw data EDA was implemented because to evaluate the relation of data with AQI and see which data is impacting AQI alot like wind speed and PM2.5 in my case. These results will help in engineering featurs which is the most important part if you want high accuracy model training and prediction.

Using EDA results and nature of project, features were engineered. As it was a time series project so time features were really important as AQI cant change instantly. Most probably last hour AQI will be next hour AQI or little bit up and down but not crazy variance unless their is fireburst as it can change AQI instantly. Features were hour, day of the week, aqi last hour, aqi last day, aqi change rate, windspeed, PM2.5, AQI

Feature pipeline was than run against the last 2 month data and transport to Features Store. MongoDB was the feature store. I wanted hopswork but it was causing problems so i used MongoDB

Four models were selected for experimentation: Random Forest, Ridge Regression, Gradient Boosting, and a Neural Network (MLP). Feature scaling was applied using a standard scaler to normalize input variables, ensuring stable optimization and fair performance comparison across models, particularly for linear and neural network-based algorithms. Each model was implemented as a multi-output regressor to predict AQI for the next 72 hours (3 days) in a single run. Engineered features were retrieved from MongoDB and used for model training. The models were evaluated using R², MAE, and RMSE metrics. Random Forest was selected as the final model because it outperformed the others across all evaluation metrics, achieving an R² score of 0.87, an MAE of 5.79, and an RMSE of 9.07.

After training, SHAP analysis was performed to identify the most important features influencing the model’s predictions. The results showed that the previous hour’s AQI was the most significant feature. This was expected, as AQI prediction is a time-series problem and AQI values do not change abruptly under normal conditions. In most cases, the AQI of the next hour is closely related to the AQI of the previous hour, with only minor fluctuations, unless an unusual event occurs—such as a fire outbreak or a gas pipeline leak—which can cause a sudden and significant change in AQI.

Chosen model which was Random forest was than transported to Dagshub Model Registry. MLflow was used for experiment tracking, model versioning and registering the latest trained model in DagsHub.

The entire system was automated using CI/CD through GitHub Actions. The feature pipeline runs every hour to ensure that the feature store always contains the latest engineered features, while the training pipeline runs daily to ensure that the model registry always holds the most recently trained model based on the latest data.

For visualizing AQI, a Streamlit dashboard was developed that fetches the latest feature data from MongoDB and the latest trained model from the DagsHub model registry. The dashboard displays the current AQI, the average AQI for the next three days, and an hourly AQI prediction graph covering a three-day horizon. Different colors are used to represent AQI levels, such as green for good air quality and red for hazardous conditions.

Link: streamlit run "C:/Users/Ali hasnain/OneDrive/Desktop/AQI_Prediction/scripts/dashboard.py"

