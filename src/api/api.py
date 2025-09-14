from flask import Flask, request, jsonify
from src.models.predict import load_model, predict_single

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Input features as per CLI
        input_features = [
            "Breed", "Age_yrs", "Lactation_Stage", "Feed_Quality",
            "Daily_Activity_hrs", "Body_Temp_C", "Heart_Rate_bpm",
            "Env_Temp_C", "Season", "Health_Status"
        ]

        # Mapping to internal feature names
        mapping = {
            "Age_yrs": "Age (yrs)",
            "Daily_Activity_hrs": "Daily_Activity (hrs)",
            "Body_Temp_C": "Body_Temp (°C)",
            "Heart_Rate_bpm": "Heart_Rate (bpm)",
            "Env_Temp_C": "Env_Temp (°C)",
        }

        for feature in input_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Create args_dict with internal names
        args_dict = {}
        for feature in input_features:
            key = mapping.get(feature, feature)
            args_dict[key] = data[feature]

        # Make prediction
        prediction = predict_single(args_dict, get_model())
        return jsonify({'predicted_milk_yield': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Cow Yield ML Model API',
        'endpoints': ['/predict (POST)'],
        'required_features': [
            "Breed", "Age_yrs", "Lactation_Stage", "Feed_Quality",
            "Daily_Activity_hrs", "Body_Temp_C", "Heart_Rate_bpm",
            "Env_Temp_C", "Season", "Health_Status"
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)