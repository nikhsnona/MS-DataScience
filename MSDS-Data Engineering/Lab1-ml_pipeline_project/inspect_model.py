import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset again to get feature names
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names

# 1. Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the importance values
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("Feature Importance:")
print(feature_importance_df)

# 2. Evaluate Model Performance
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the scaler to the test data
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# 3. Print Model Structure
print("\nModel Structure:")
print(model)

# 4. Inspect Individual Trees (Optional)
# Print the first decision tree in the Random Forest
tree = model.estimators_[0]
print("\nFirst Decision Tree in the Forest:")
print(tree)
