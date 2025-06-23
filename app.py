from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model and label encoder
model = joblib.load("universal_parameter_classifier.pkl")
label_encoder = joblib.load("universal_label_encoder.pkl")

# Define expected parameter columns for one-hot encoding
param_cols = ["Parameter_Hemoglobin", "Parameter_WBC", "Parameter_Platelet"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract input values
        param = data.get("parameter")  # e.g., "Hemoglobin"
        value = float(data.get("value"))
        age = int(data.get("age"))
        gender = 1 if data.get("gender") == "M" else 0

        # One-hot encode the parameter
        param_vector = [1 if f"Parameter_{param}" == col else 0 for col in param_cols]

        # Combine all features
        input_data = [[value, age, gender] + param_vector]

        # Make prediction
        prediction = model.predict(input_data)
        label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"result": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
