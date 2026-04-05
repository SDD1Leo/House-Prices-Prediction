# app.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

from train import train_model

app = Flask(__name__)

# Load model
if not os.path.exists("model.pkl"):
    print("model.pkl not found! Training the model...")
    train_model()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "House Price Prediction API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input into dataframe
        df = pd.DataFrame([data])

        # Feature engineering (IMPORTANT: same as training)
        df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
        df['household_rooms'] = df['total_rooms'] / df['households']

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