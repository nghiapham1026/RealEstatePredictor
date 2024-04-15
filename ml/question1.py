import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data_path = '../realtor-data_small.csv'
house_data = pd.read_csv(data_path)

# Define features and target
numerical_features = ['bed', 'bath', 'acre_lot', 'house_size']
categorical_features = ['city']
features = numerical_features + categorical_features
target = 'price'

# Clean the data: Remove rows with missing target
house_data = house_data[house_data[target].notna()]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    house_data[features], house_data[target], test_size=0.2, random_state=0)

# Define preprocessing for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=0)
}

# Evaluate models using cross-validation
cv_results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_results[name] = -scores.mean()  # Convert to positive MSE

# Select the best model based on cross-validation MSE
best_model_name = min(cv_results, key=cv_results.get)
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', models[best_model_name])])

# Train the best model on the full training set and evaluate it on the test set
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Print results
print(f"Best model: {best_model_name} with test MSE: {mse_test} and R^2: {r2_test}")
