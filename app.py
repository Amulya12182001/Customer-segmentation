from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
app = Flask(__name__, template_folder="src", static_folder="src")

# Load the saved model and scaler
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    try:
        recency = float(request.form['Recency'])
        frequency = float(request.form['Frequency'])
        monetary = float(request.form['Monetary'])
    except ValueError:
        return "Invalid input! Please enter numeric values."

    # Prepare the input data and scale it
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    scaled_input = scaler.transform(input_data)

    # Predict the cluster
    cluster = kmeans.predict(scaled_input)

    # Display the result
    return render_template('index.html', cluster=cluster[0], recency=recency, frequency=frequency, monetary=monetary)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
