from flask import Flask, request, jsonify
import pickle
import numpy as np
import time
import datetime
import joblib
import logging
import random
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all domains to access your Flask app

# Load trained model and scaler once when the app starts
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    exit(1)

# Setup logging
logging.basicConfig(filename="api_requests.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Store response times for plotting (for response time vs iteration)
iteration_data = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = datetime.datetime.now()

        data = request.json["features"]
        logging.info(f"Received Features: {data}")
        data_array = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data_array)
        logging.info("Data scaled successfully.")

        prediction = model.predict(data_scaled)[0]
        logging.info(f"Prediction: {prediction}")

        flower_type = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}[prediction]
        model_type = type(model).__name__
        confidence = model.predict_proba(data_scaled).max() * 100  # Confidence in percentage

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Store response time for the iteration
        iteration_data.append(processing_time)

        logging.info(f"Timestamp: {timestamp}, Flower Type: {flower_type}, Confidence: {confidence:.2f}%, Processing Time: {processing_time:.4f} sec")

        return jsonify({
            "timestamp": timestamp,
            "prediction": flower_type,
            "flower_type": flower_type,
            "model_type": model_type,
            "confidence_level": round(confidence, 2),
            "processing_time": processing_time,  # Include processing time
            "iteration_data": iteration_data  # Send the data for plotting
        })

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/random_prediction", methods=["GET"])
def random_prediction():
    try:
        random_features = [random.uniform(4.0, 8.0), random.uniform(2.0, 4.5), random.uniform(1.0, 7.0), random.uniform(0.1, 2.5)]
        logging.info(f"Generated Random Features: {random_features}")

        data_array = np.array(random_features).reshape(1, -1)
        data_scaled = scaler.transform(data_array)
        logging.info("Random data scaled successfully.")

        prediction = model.predict(data_scaled)[0]
        logging.info(f"Random Prediction: {prediction}")

        flower_type = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}[prediction]
        model_type = type(model).__name__
        confidence = model.predict_proba(data_scaled).max() * 100

        return jsonify({
            "prediction": flower_type,
            "flower_type": flower_type,
            "model_type": model_type,
            "confidence_level": round(confidence, 2)
        })

    except Exception as e:
        logging.error(f"Error in /random_prediction: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
