# app.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "House Price Prediction API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert input into dataframe
    df = pd.DataFrame([data])

    # Feature engineering (IMPORTANT: same as training)
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']

    prediction = model.predict(df)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)