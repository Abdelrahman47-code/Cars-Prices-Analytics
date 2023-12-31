from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model
with open('rf_model_best.pkl', 'rb') as model_file:
    rf_model = joblib.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data (values from sliders and dropdowns)
    length = float(request.json['length'])
    width = float(request.json['width'])
    height = float(request.json['height'])
    wheel_base = float(request.json['wheel_base'])
    cargo_volume = float(request.json['cargo_volume'])
    max_torque = float(request.json['max_torque'])
    seats = float(request.json['seats'])
    transmission_encoded = float(request.json['transmission'])
    fuel_encoded = float(request.json['fuel'])
    gear_box_encoded = float(request.json['gear_box'])

    # Perform prediction using your model
    prediction_result = rf_model.predict([[length, width, height, wheel_base, cargo_volume, max_torque, seats,
                                           transmission_encoded, fuel_encoded, gear_box_encoded]])[0]

    # Return the predicted price as JSON
    return jsonify({'predicted_price': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
