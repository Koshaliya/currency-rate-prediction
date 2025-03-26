from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import datetime
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from utils import load_model_and_predict  # Put your actual prediction logic in utils.py

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Currency Exchange Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1. Required input
        currency = data.get("currency")
        if not currency:
            return jsonify({"error": "Currency parameter is required."}), 400

        currency_key = f"GBP_{currency.upper()}"

        # 2. Optional dates
        start_date = data.get("start_date", None)
        end_date = data.get("end_date", None)

        # 3. File paths
        model_path = f"models/model_{currency_key}.h5"
        scaler_path = f"models/scaler_{currency_key}.pkl"

        # 4. Check if model and scaler exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"Model or scaler for {currency_key} not found."}), 404

        # 5. Predict and return
        predictions = load_model_and_predict(currency_key, start_date, end_date)
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
