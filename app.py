from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from pdf2image import convert_from_bytes
import re
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load ML model and label encoder
model = joblib.load("universal_parameter_classifier.pkl")
label_encoder = joblib.load("universal_label_encoder.pkl")

# Define expected one-hot encoded parameter columns
param_cols = ["Parameter_Hemoglobin", "Parameter_WBC", "Parameter_Platelet Count", "Parameter_RBC", "Parameter_PCV", "Parameter_MCH", "Parameter_MCV", "Parameter_MCHC"]

@app.route('/', methods=['GET'])
def home():
    return "MediScan AI Backend is Running."

@app.route('/extract', methods=['POST'])
def extract_pdf():
    file = request.files.get('file')
    age = int(request.form.get("age", 25))
    gender = request.form.get("gender", "F")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        images = convert_from_bytes(file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])

        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        parameters_to_find = {
            "Hemoglobin": r"[\d.]+",
            "Platelet Count": r"[\d.,]+",
            "WBC": r"[\d.,]+",
            "RBC": r"[\d.,]+",
            "PCV": r"[\d.]+",
            "MCH": r"[\d.]+",
            "MCV": r"[\d.]+",
            "MCHC": r"[\d.]+",
        }

        predictions = []

        for idx, line in enumerate(lines):
            for param, pattern in parameters_to_find.items():
                if param.lower() in line.lower():
                    match = re.search(pattern, line)
                    value = None
                    if match:
                        value = match.group()
                    else:
                        if idx + 1 < len(lines):
                            match = re.search(pattern, lines[idx + 1])
                            if match:
                                value = match.group()

                    if value:
                        try:
                            val = float(value.replace(",", ""))
                            gender_enc = 1 if gender.upper() == "M" else 0
                            param_vector = [1 if f"Parameter_{param}" == col else 0 for col in param_cols]
                            input_data = [[val, age, gender_enc] + param_vector]
                            prediction = model.predict(input_data)
                            label = label_encoder.inverse_transform(prediction)[0]
                        except Exception as e:
                            label = "-"

                        predictions.append({
                            "parameter": param,
                            "value": value,
                            "status": label
                        })

        return jsonify({
            "status": "completed",
            "results": predictions
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
