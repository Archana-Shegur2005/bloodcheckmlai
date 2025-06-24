from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import re
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
        # Use PyMuPDF to extract text from PDF
        doc = fitz.open("pdf", file.read())
        text = "\n".join([page.get_text() for page in doc])
        print("📄 Extracted Text:\n", text)

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
                            # One-hot encode for model
                            input_vector = np.zeros(len(model.feature_names_in_))
                            for i, name in enumerate(model.feature_names_in_):
                                if name == f"Parameter_{param}":
                                    input_vector[i] = 1
                                elif name == "Value":
                                    input_vector[i] = clean_value
                            pred = model.predict([input_vector])[0]
                            status = label_encoder.inverse_transform([pred])[0]

                            predictions.append({
                                "parameter": param,
                                "value": str(clean_value),
                                "status": status
                            })
                    except Exception as e:
                        print(f"⚠️ Prediction failed for {param}: {e}")
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
        print("❌ PDF processing error:", e)
        return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == "__main__":
    app.run(debug=True)
