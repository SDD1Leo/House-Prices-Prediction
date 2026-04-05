# app.py

from flask import Flask, request, jsonify
from flasgger import Swagger
import pickle
import pandas as pd
import os

from train import train_model

app = Flask(__name__)
swagger = Swagger(app)

# Load model
if not os.path.exists("model.pkl"):
    print("model.pkl not found! Training the model...")
    train_model()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    """
    Home Endpoint
    ---
    responses:
      200:
        description: Returns a welcome message
    """
    return "House Price Prediction API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict California House Price
    ---
    parameters:
      - in: body
        name: body
        schema:
          id: HouseData
          required:
            - longitude
            - latitude
            - housing_median_age
            - total_rooms
            - total_bedrooms
            - population
            - households
            - median_income
          properties:
            longitude: {type: number, example: -122.23}
            latitude: {type: number, example: 37.88}
            housing_median_age: {type: number, example: 41.0}
            total_rooms: {type: number, example: 880.0}
            total_bedrooms: {type: number, example: 129.0}
            population: {type: number, example: 322.0}
            households: {type: number, example: 126.0}
            median_income: {type: number, example: 8.3252}
    responses:
      200:
        description: Predicted price
      400:
        description: Invalid or missing input
      500:
        description: Server error
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input into dataframe
        df = pd.DataFrame([data])

        # Feature engineering (IMPORTANT: same as training)
        df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
        df['household_rooms'] = df['total_rooms'] / df['households']

        # Enforce exact column order required by the model
        expected_cols = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "bedroom_ratio", "household_rooms"
        ]
        df = df[expected_cols]

        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": float(prediction)
        })
    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)