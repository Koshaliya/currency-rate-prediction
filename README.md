# Currency Exchange Rate Prediction

This project focuses on building and evaluating machine learning models to predict currency exchange rates from GBP to EUR, HKD, USD, and JPY. The notebook implements and compares three different models: LSTM (Deep Learning), ARIMA (Time Series), and Random Forest (Machine Learning) to identify the best-performing approach.

## Models Used
1. LSTM (Long Short-Term Memory)
Captures time-based trends using deep learning.
Built using TensorFlow/Keras.

2. ARIMA
Traditional time-series forecasting model.
Suitable for univariate data with trend/seasonality.

3. Random Forest
Tree-based ensemble model.
Handles nonlinear patterns effectively.

## Dataset
Historical currency exchange rate data collected from Bloomberg service of UWE.
Features include timestamped exchange rates for GBP to multiple currencies.
Data preprocessing includes missing value handling, scaling, and feature engineering.

