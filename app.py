from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow.keras.models as keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
model = keras.load_model("cnn_model.h5")
print("Model loaded successfully.")

# Load the label encoder to decode crop names
data = pd.read_csv("crop_data1.csv")  # Same CSV used for training
label_encoder = {i: label for i, label in enumerate(data['label'].unique())}

# Load the scaler used for data normalization
scaler = StandardScaler()
scaler.fit(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])

# Home route
@app.route('/')
def home():
    return render_template("index.html")  # A simple HTML form for user input

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Preprocess input for the CNN model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)
        reshaped_input = scaled_input.reshape(-1, 7, 1, 1)  # (samples, features, height, channels)

        # Make prediction
        prediction = model.predict(reshaped_input)
        crop_index = np.argmax(prediction)
        crop_name = label_encoder[crop_index]

        # Return the recommendation
        return render_template("result.html", crop=crop_name)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(port=8080)






