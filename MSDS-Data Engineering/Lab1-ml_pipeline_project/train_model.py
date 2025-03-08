import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Apply scaling to the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y)

# Save the model and the scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved as 'model.pkl' and 'scaler.pkl'")
