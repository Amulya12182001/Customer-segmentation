import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load the saved model and scaler
with open("kmeans_model.pkl", 'rb') as f:
    kmeans = pickle.load(f)
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
# Streamlit App
def main():
    # Title of the app
    st.title("Customer Segmentation with KMeans Clustering")
    # Create input fields for user data
    recency = st.number_input("Recency", min_value=0, step=1)
    frequency = st.number_input("Frequency", min_value=0, step=1)
    monetary = st.number_input("Monetary", min_value=0, step=1)
    # Button to predict cluster
    if st.button("Predict Cluster"):
        try:
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
            st.write(f"The customer belongs to Cluster {cluster[0]}")
            st.write(f"Recency: {recency}, Frequency: {frequency}, Monetary: {monetary}")
        except ValueError:
            st.error("Invalid input! Please enter numeric values.")
if __name__ == "__main__":
    main()
