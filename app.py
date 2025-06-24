from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from pdf2image import convert_from_bytes
import re
import joblib
import pandas as pd

# Load ML model and encoder
model = joblib.load("universal_parameter_classifier.pkl")
label_encoder = joblib.load("universal_label_encoder.pkl")

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/extract', methods=['POST'])
def extract_from_pdf():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        images = convert_from_bytes(file.read())
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        print("🔍 OCR Extracted Text:\n", text)
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

        results = []

        for idx, line in enumerate(lines):
            for param, pattern in parameters_to_find.items():
                if param.lower() in line.lower():
                    match = re.search(pattern, line)
                    value = None
                    if match:
                        value = match.group()
                    elif idx + 1 < len(lines):
                        match = re.search(pattern, lines[idx + 1])
                        if match:
                            value = match.group()

                    # Validate and parse value
                    try:
                        float_value = float(value.replace(",", ""))
                    except:
                        continue  # skip invalid

                    # Prepare features for prediction
                    feature_row = pd.DataFrame({"Parameter": [param]})
                    X = pd.get_dummies(feature_row)
                    for col in model.feature_names_in_:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[model.feature_names_in_]
                    X["Value"] = float_value

                    # Predict
                    try:
                        pred_encoded = model.predict(X)[0]
                        status = label_encoder.inverse_transform([pred_encoded])[0]
                    except Exception as e:
                        print(f"Prediction error for {param}:", e)
                        status = "-"

                    results.append({
                        "parameter": param,
                        "value": str(float_value),
                        "status": status
                    })

        return jsonify({"status": "completed", "results": results})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == '__main__':
    app.run(debug=True)
