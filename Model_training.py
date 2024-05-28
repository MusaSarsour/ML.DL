import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load the dataset
stock_data = pd.read_csv("Stock.csv")

# Select relevant features for the model
features = ['adjHigh', 'adjLow', 'adjOpen', 'adjVolume']
target = 'adjClose'

# Drop rows with missing values
stock_data = stock_data.dropna(subset=features + [target])

# Split the data into training and testing sets
X = stock_data[features]
y = stock_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with polynomial features and ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

# Define the parameter grid for Ridge Regression
param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0],
    'poly_features__degree': [1, 2, 3]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'improved_stock_model.pkl')
