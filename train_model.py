import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
csv_file = "blood_parameter_dataset_by_row.csv"  # Make sure this file is in the same folder
print("Loading dataset from:", csv_file)
df = pd.read_csv(csv_file)

# Encode gender
print("Encoding gender...")
df['GenderEncoded'] = df['Gender'].map({'F': 0, 'M': 1})

# One-hot encode 'Parameter'
print("One-hot encoding parameters...")
df_encoded = pd.get_dummies(df, columns=['Parameter'])

# Encode target labels (Low, Normal, High)
print("Encoding labels...")
le = LabelEncoder()
df_encoded['LabelEncoded'] = le.fit_transform(df_encoded['Label'])

# Define feature columns
print("Preparing features...")
feature_cols = ['Value', 'Age', 'GenderEncoded'] + [col for col in df_encoded.columns if col.startswith("Parameter_")]
X = df_encoded[feature_cols]
y = df_encoded['LabelEncoded']

# Train-test split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder
print("Saving model and label encoder...")
joblib.dump(model, "universal_parameter_classifier.pkl")
joblib.dump(le, "universal_label_encoder.pkl")

print("✅ Training complete. Model saved as .pkl files")
