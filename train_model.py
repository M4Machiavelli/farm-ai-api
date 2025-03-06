import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset (Replace with real data if available)
data_size = 1000
crop_types = ["Wheat", "Rice", "Corn", "Soybean", "Tomato"]
actions = ["Irrigate", "Fertilize", "Adjust pH", "No Action"]

df = pd.DataFrame({
    "humidity": np.random.uniform(40, 90, data_size),
    "temperature": np.random.uniform(15, 40, data_size),
    "water_level": np.random.uniform(20, 80, data_size),
    "soil_moisture": np.random.uniform(10, 60, data_size),
    "acidity": np.random.uniform(5.5, 7.5, data_size),
    "crop_type": np.random.choice(crop_types, data_size),
    "recommendation": np.random.choice(actions, data_size)
})

# Encode categorical variables
label_enc_crop = LabelEncoder()
label_enc_action = LabelEncoder()

df["crop_type"] = label_enc_crop.fit_transform(df["crop_type"])
df["recommendation"] = label_enc_action.fit_transform(df["recommendation"])

# Define features & labels
X = df.drop(columns=["recommendation"])  # Features
y = df["recommendation"]  # Target variable

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest model with optimized hyperparameters
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate accuracy and other metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model and encoders
joblib.dump(model, "farm_ai_model.pkl")
joblib.dump(label_enc_crop, "crop_type_encoder.pkl")
joblib.dump(label_enc_action, "action_encoder.pkl")

print("Model and encoders saved successfully!")

