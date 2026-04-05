# train.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset (keep CSV locally in project folder)
train_data = pd.read_csv("california_housing_train.csv")

# Feature engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

# Split features and target
X = train_data.drop(['median_house_value'], axis=1)
y = train_data['median_house_value']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved ✅")