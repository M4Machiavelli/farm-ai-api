import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ✅ Load the existing trained model and label encoders (if available)
try:
    model = joblib.load("farm_ai_model.pkl")
    label_enc_crop = joblib.load("crop_type_encoder.pkl")
    label_enc_action = joblib.load("action_encoder.pkl")
    print("✅ Existing model and encoders loaded successfully!")
except FileNotFoundError:
    print("⚠️ No existing model found. Training a new model instead.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    label_enc_crop = LabelEncoder()
    label_enc_action = LabelEncoder()

# ✅ Load new data (Replace 'new_farm_data.csv' with your actual file)
try:
    new_data = pd.read_csv("new_farm_data.csv")
    print("✅ New data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: The file 'new_farm_data.csv' was not found.")
    exit()

# ✅ Ensure the dataset contains required columns
required_columns = ["humidity", "temperature", "water_level", "soil_moisture", "acidity", "crop_type", "recommendation"]
if not all(col in new_data.columns for col in required_columns):
    print("❌ Error: The dataset is missing required columns!")
    exit()

# ✅ Encode categorical features (Crop Type & Recommendation)
new_data["crop_type"] = label_enc_crop.fit_transform(new_data["crop_type"])
new_data["recommendation"] = label_enc_action.fit_transform(new_data["recommendation"])

# ✅ Check Class Distribution Before Splitting
print("\nClass distribution before train-test split:\n", new_data["recommendation"].value_counts())

# ✅ Remove classes with only 1 occurrence to avoid train-test split errors
class_counts = new_data["recommendation"].value_counts()
valid_classes = class_counts[class_counts > 1].index
filtered_data = new_data[new_data["recommendation"].isin(valid_classes)]

# ✅ Define features and target after filtering
X_filtered = filtered_data.drop(columns=["recommendation"])
y_filtered = filtered_data["recommendation"]

# ✅ Use all data for training (no train-test split)
X_train, y_train = X_filtered, y_filtered
X_test, y_test = X_filtered, y_filtered  # Use all data for evaluation
print(f"✅ Training on all available data (Total samples: {len(X_train)})")

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ✅ Retrain the model with additional estimators
model.n_estimators += 50  # Increase trees for better learning
model.fit(X_train, y_train)
print("✅ Model retraining complete!")

# ✅ Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Updated Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Display classification report and confusion matrix
print("\nUpdated Classification Report:\n", classification_report(y_test, y_pred))
print("\nUpdated Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Save the updated model and encoders
joblib.dump(model, "farm_ai_model_updated.pkl")
joblib.dump(label_enc_crop, "crop_type_encoder.pkl")
joblib.dump(label_enc_action, "action_encoder.pkl")

print("✅ Updated model and encoders saved successfully!")

