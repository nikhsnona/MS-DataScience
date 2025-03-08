# Iris Flower Prediction ML Pipeline

This project uses machine learning to predict the species of Iris flowers based on input features like sepal and petal dimensions. It involves training a model, deploying it with Flask, and allowing users to interact with it via a simple web interface.

## Project Structure

```
ML_PIPELINE_PROJECT/
├── train_model.py        # Script to train the machine learning model
├── server.py             # Flask API to serve the model and handle requests
├── client.py             # Simulate client requests to the API
├── inspect_model.py      # Script for model evaluation (performance, feature importance)
├── index.html            # Frontend interface to interact with the model
├── static/               # Static assets (CSS, JavaScript, etc.)
│   ├── style.css         # CSS styling for the frontend
│   └── script.js         # JavaScript to interact with Flask API
└── requirements.txt      # List of dependencies
```

## Installation

### Prerequisites

- Python 3.6+
- Required libraries can be installed using `requirements.txt`.

Run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- Flask: Web framework for serving the model
- Scikit-learn: For machine learning model (Random Forest)
- Joblib: For saving/loading model and scaler
- Matplotlib: For plotting response times
- Requests: For client-server interactions
- Chart.js: For frontend visualization of response time

## Setup and Usage

### 1. **Train the Model**

Run `train_model.py` to train the Random Forest model using the Iris dataset. This will generate `model.pkl` (the trained model) and `scaler.pkl` (the feature scaler).

```bash
python train_model.py
```

### 2. **Run the Flask Server**

Start the Flask API by running `server.py`. This will expose two main endpoints:

- `/predict`: Accepts flower measurements and returns the predicted species.
- `/random_prediction`: Generates random flower measurements and predicts the species.

```bash
python server.py
```

The API will be available at `http://127.0.0.1:5000`.

### 3. **Frontend Interface**

- Open `index.html` in your browser.
- Enter flower measurements (sepal length, sepal width, petal length, petal width) to get a prediction.
- The page also includes a chart to display response times of predictions.

### 4. **Inspect the Model**

Use `inspect_model.py` to evaluate the model's performance, feature importance, and accuracy.

```bash
python inspect_model.py
```

### 5. **Simulate Client Requests**

The `client.py` script simulates multiple requests to the Flask server and plots the response times over iterations.

```bash
python client.py
```