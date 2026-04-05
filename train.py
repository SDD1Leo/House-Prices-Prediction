import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

def train_model():
    print("Loading data...")
    df = pd.read_csv("california_housing_train.csv")
    
    # Feature engineering (must match prediction exactly)
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    
    # Handle any missing values just in case
    df = df.fillna(0)
    
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Saving model to model.pkl...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()