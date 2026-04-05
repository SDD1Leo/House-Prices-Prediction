# Housing Price Prediction

This project involves building a machine learning pipeline to predict median house values in California using various regression techniques.

## Project Overview
- **Objective**: Predict `median_house_value` based on census data.
- **Dataset**: California Housing dataset (standard Scikit-Learn / Colab sample data).

## Workflow
### 1. Data Acquisition & EDA
- Data was loaded from `california_housing_train.csv`.
- Exploratory Data Analysis (EDA) was performed using `matplotlib` and `seaborn` to visualize geographical distributions and feature correlations.

### 2. Feature Engineering
To improve model performance, the following ratios were calculated:
- **Bedroom Ratio**: `total_bedrooms` / `total_rooms`
- **Household Rooms**: `total_rooms` / `households`

### 3. Model Development
We compared multiple models to find the best fit:
- **Linear Regression**: Served as the baseline model (R² ≈ 0.63).
- **Random Forest Regressor**: Significantly improved accuracy (R² ≈ 0.80).
- **XGBoost**: Implemented with GPU acceleration to target higher precision goals.

### 4. Hyperparameter Tuning
Used `GridSearchCV` to optimize model parameters, focusing on `n_estimators`, `max_depth`, and `learning_rate` for XGBoost.

## Interactive Tools
- Included a **Colab Forms** interface for real-time inference. Users can adjust sliders for coordinates, income, and age to see immediate price predictions.

## Results
The Random Forest model achieved an R² score of ~0.80 on the test set. Further optimizations with XGBoost were explored to reach the target accuracy threshold.
