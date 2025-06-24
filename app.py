from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import fitz  # PyMuPDF
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("universal_parameter_classifier.pkl")
label_encoder = joblib.load("universal_label_encoder.pkl")

reference_parameters = {
    "Hemoglobin": r"[\d.]+",
    "Platelet Count": r"[\d.,]+",
    "WBC": r"[\d.,]+",
    "RBC": r"[\d.,]+",
    "PCV": r"[\d.]+",
    "MCH": r"[\d.]+",
    "MCV": r"[\d.]+",
    "MCHC": r"[\d.]+",
}

@app.route("/extract", methods=["POST"])
def extract_parameters():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Extract raw text from the uploaded PDF
        pdf_data = file.read()
        doc = fitz.open("pdf", pdf_data)
        text = "\n".join([page.get_text() for page in doc])
        print("🔍 PDF Text:\n", text)

        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        predictions = []

        for idx, line in enumerate(lines):
            for param, pattern in reference_parameters.items():
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

                    try:
                        if value:
                            clean_value = float(value.replace(",", "").replace(" ", ""))
                            # Use ML model
                            input_df = {f"Parameter_{param}": [1], "Value": [clean_value]}
                            X = model.feature_names_in_
                            vector = np.zeros(len(X))
                            for i, name in enumerate(X):
                                if name == f"Parameter_{param}":
                                    vector[i] = 1
                                elif name == "Value":
                                    vector[i] = clean_value
                            prediction = model.predict([vector])[0]
                            status = label_encoder.inverse_transform([prediction])[0]

                            predictions.append({
                                "parameter": param,
                                "value": str(clean_value),
                                "status": status
                            })
                    except Exception as e:
                        print(f"Prediction failed for {param} with value {value}: {e}")
                        predictions.append({
                            "parameter": param,
                            "value": value if value else "-",
                            "status": "-"
                        })

        return jsonify({
            "status": "completed",
            "results": predictions
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == "__main__":
    app.run(debug=True)
