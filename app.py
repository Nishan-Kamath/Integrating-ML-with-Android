"""from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model (pipeline)
model = joblib.load('placement_model.pkl')

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve JSON data from the request
        data = request.get_json()

        # Extract the individual fields
        cgpa = data.get('cgpa')
        iq = data.get('iq')
        profile_score = data.get('profile_score')

        # Check for missing inputs
        if cgpa is None or iq is None or profile_score is None:
            return jsonify({'error': 'All input fields (cgpa, iq, profile_score) are required.'}), 400

        # Convert inputs to floats
        try:
            cgpa = float(cgpa)
            iq = float(iq)
            profile_score = float(profile_score)
        except ValueError:
            return jsonify({'error': 'Input values must be numbers.'}), 400

        # Prepare the input for prediction
        input_query = np.array([[cgpa, iq, profile_score]])
        result = model.predict(input_query)

        return jsonify({'placement': str(result[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ =="__main__":
    app.run(debug=True)"""
from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Load the pre-trained model (pipeline)
#model = joblib.load('placement_model.pkl')

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve query parameters from the URL
        cgpa = request.args.get('cgpa')
        iq = request.args.get('iq')
        profile_score = request.args.get('profile_score')

        # Check for missing inputs
        if cgpa is None or iq is None or profile_score is None:
            return jsonify({'error': 'All input fields (cgpa, iq, profile_score) are required.'}), 400

        # Convert inputs to floats
        try:
            cgpa = float(cgpa)
            iq = float(iq)
            profile_score = float(profile_score)
        except ValueError:
            return jsonify({'error': 'Input values must be numbers.'}), 400

        # Prepare the input for prediction
        input_query = np.array([[cgpa, iq, profile_score]])
        if(cgpa>=7.0 and iq>=100 and profile_score>=100):
            result = 1
        else:
            result = 0

        return jsonify({'placement': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

