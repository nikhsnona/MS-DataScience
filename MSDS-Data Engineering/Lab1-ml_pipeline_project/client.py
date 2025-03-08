import requests
import time
import matplotlib.pyplot as plt

# URL of the Flask server endpoints
predict_url = 'http://127.0.0.1:5000/predict'
random_url = 'http://127.0.0.1:5000/random_prediction'

# List to store response times for graphing
response_times_predict = []  # For predict response times
response_times_random = []   # For random prediction response times

# Function to perform prediction
def predict(features):
    try:
        start_time = time.time()  # Start timer

        # Make POST request to the Flask server
        response = requests.post(predict_url, json={"features": features})

        if response.status_code == 200:
            # Record the response time
            end_time = time.time()
            response_time = end_time - start_time
            response_times_predict.append(response_time)

            data = response.json()
            print(f"Prediction: {data['prediction']}")
            print(f"Flower Type: {data['flower_type']}")
            print(f"Model Type: {data['model_type']}")
            print(f"Confidence: {data['confidence_level']}%")
            print(f"Response Time: {response_time:.4f} seconds")
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error occurred during prediction: {e}")

# Function to perform random prediction
def random_prediction():
    try:
        start_time = time.time()  # Start timer

        # Make GET request to the Flask server
        response = requests.get(random_url)

        if response.status_code == 200:
            # Record the response time
            end_time = time.time()
            response_time = end_time - start_time
            response_times_random.append(response_time)

            data = response.json()
            print(f"Random Prediction: {data['prediction']}")
            print(f"Flower Type: {data['flower_type']}")
            print(f"Model Type: {data['model_type']}")
            print(f"Confidence: {data['confidence_level']}%")
            print(f"Response Time: {response_time:.4f} seconds")
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error occurred during random prediction: {e}")

# Main function to send predictions multiple times
def main():
    # For example, make 10 predictions and 10 random predictions
    iterations = 10
    for i in range(iterations):
        print(f"Iteration {i + 1}:")
        # For prediction, we will use some example features
        features = [5.1, 3.5, 1.4, 0.2]  # Example features for Iris Setosa
        predict(features)
        # Perform random prediction every iteration (can be changed)
        random_prediction()

    # Plotting the Response Time vs Iteration graph for both
    plt.plot(range(1, len(response_times_predict) + 1), response_times_predict, marker='o', linestyle='-', color='b', label='Prediction Response Time')
    plt.plot(range(1, len(response_times_random) + 1), response_times_random, marker='x', linestyle='--', color='r', label='Random Prediction Response Time')

    plt.title('Response Time vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Response Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()  # This will display the graph

if __name__ == "__main__":
    main()
