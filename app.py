from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
with open('crop_model.pkl', 'rb') as model_file:
    crop_model = pickle.load(model_file)
print("✅ Crop Model loaded successfully!")

# Load the trained scaler
with open('crop_scaler.pkl', 'rb') as scaler_file:
    crop_scaler = pickle.load(scaler_file)

# Define feature column names manually
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Initialize LabelEncoder with known classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon', 'wheat'
])

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array([[  
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])

        # Convert input to DataFrame with feature names
        features_df = pd.DataFrame(features, columns=feature_columns)

        # Scale input
        features_scaled = crop_scaler.transform(features_df)

        # Predict crop
        prediction = crop_model.predict(features_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'recommended_crop': predicted_crop})

    except Exception as e:
        return jsonify({'error': str(e)})


# # Load the trained model
# with open('yield_model.pkl', 'rb') as model_file:
#     yield_model = pickle.load(model_file)
# print("✅ Yield Model loaded successfully!")

# # Load the trained scaler
# with open('yield_scaler.pkl', 'rb') as scaler_file:
#     yield_scaler = pickle.load(scaler_file)
# # Manually recreate label encoders
# season_labels = ["Kharif", "Rabi", "Summer", "Whole Year", "Winter", "Autumn"]
# crop_labels = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
#                'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
#                'grapes', 'watermelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton',
#                'jute', 'coffee']

# # Create label encoders and fit them with the known categories
# season_encoder = LabelEncoder()
# season_encoder.fit(season_labels)

# crop_encoder = LabelEncoder()
# crop_encoder.fit(crop_labels)

# @app.route('/predict_yield', methods=['POST'])
# def predict_yield():
#     try:
#         data = request.json
#         crop_year = int(data['Crop_Year'])
#         season = data['Season'].strip()
#         crop = data['Crop'].strip().lower()
#         area = float(data['Area'])

#         # Validate inputs
#         if season not in season_labels:
#             return jsonify({"error": f"Invalid season '{season}'. Available: {season_labels}"}), 400
#         if crop not in crop_labels:
#             return jsonify({"error": f"Invalid crop '{crop}'. Available: {crop_labels}"}), 400

#         # Encode categorical variables
#         season_encoded = season_encoder.transform([season])[0]
#         crop_encoded = crop_encoder.transform([crop])[0]

#         # Prepare input data
#         input_data = np.array([[crop_year, season_encoded, crop_encoded, area]])

#         # Standardize input
#         input_scaled = yield_scaler.transform(input_data)

#         # Predict yield
#         predicted_yield = yield_model.predict(input_scaled)

#         return jsonify({
#             "Predicted_Yield": round(predicted_yield[0], 2)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render runs on port 10000