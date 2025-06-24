from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import re
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model and label encoder
model = joblib.load("universal_parameter_classifier.pkl")

@app.route('/extract', methods=['POST'])
def extract():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Step 1: Convert PDF to text using PyMuPDF
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()

        print("🔍 OCR Extracted Text:\n", text)

        # Step 2: Split into lines and clean
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Step 3: Define patterns for known parameters
        parameters_to_find = {
            "Hemoglobin": r"\d+\.?\d*",
            "Platelet Count": r"\d+\.?\d*",
            "WBC": r"\d+\.?\d*",
            "RBC": r"\d+\.?\d*",
            "PCV": r"\d+\.?\d*",
            "MCH": r"\d+\.?\d*",
            "MCV": r"\d+\.?\d*",
            "MCHC": r"\d+\.?\d*",
        }

        predictions = []
        added_params = set()

        for idx, line in enumerate(lines):
            for param, pattern in parameters_to_find.items():
                if param.lower() in line.lower() and param not in added_params:
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
                        clean_val = value.replace(",", "").replace(" ", "").strip()
                        try:
                            val_float = float(clean_val)
                        except:
                            continue  # skip invalid numbers like ',' or '.'

                        # ML Prediction
                        try:
                            features = pd.DataFrame([{
                                "Parameter": param,
                                "Value": val_float
                            }])
                            status = model.predict(features)[0]
                        except Exception as e:
                            print(f"Prediction failed for {param} with value {val_float}: {e}")
                            status = "-"

                        predictions.append({
                            "parameter": param,
                            "value": str(val_float),
                            "status": status
                        })
                        added_params.add(param)

        return jsonify({
            "status": "completed",
            "results": predictions
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to process PDF"}), 500


if __name__ == '__main__':
    app.run(debug=True)
