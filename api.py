from fastapi import FastAPI
import joblib
import numpy as np

# Load the trained AI model and encoders
model = joblib.load("farm_ai_model_updated.pkl")
crop_type_encoder = joblib.load("crop_type_encoder.pkl")
action_encoder = joblib.load("action_encoder.pkl")

# Initialize FastAPI
app = FastAPI()

# Define API endpoint for predictions
@app.post("/predict/")
def predict(
    humidity: float,
    temperature: float,
    water_level: float,
    soil_moisture: float,
    acidity: float,
    crop_type: str
):
    try:
        # Convert crop type to numerical format
        if crop_type not in crop_type_encoder.classes_:
            return {"error": f"Invalid crop type. Use one of: {', '.join(crop_type_encoder.classes_)}"}

        crop_index = crop_type_encoder.transform([crop_type])[0]

        # Prepare input data for model
        input_data = np.array([[humidity, temperature, water_level, soil_moisture, acidity, crop_index]])

        # Predict recommendation
        prediction = model.predict(input_data)[0]
        recommendation = action_encoder.inverse_transform([prediction])[0]

        return {"recommendation": recommendation}

    except Exception as e:
        return {"error": str(e)}


