import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset from CSV file
df = pd.read_csv("dataset/data.csv")  # Ensure this file is in the same folder as this script

# Check required columns exist
required_columns = {"Parameter", "Value", "Status"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

# Encode 'Status' labels
label_encoder = LabelEncoder()
df["Status_Encoded"] = label_encoder.fit_transform(df["Status"])

# One-hot encode 'Parameter'
X = pd.get_dummies(df[["Parameter"]])
X["Value"] = df["Value"]
y = df["Status_Encoded"]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model and label encoder
joblib.dump(model, "universal_parameter_classifier.pkl")
joblib.dump(label_encoder, "universal_label_encoder.pkl")

print("\u2705 Model and encoders saved successfully using CSV dataset.")
