import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data_path = '../realtor-data_small.csv'
house_data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Define features and target
numerical_features = ['house_size']  # Focusing on house size primarily
categorical_features = ['city']  # Correlating categorical feature
features = numerical_features + categorical_features
target = 'price'

# Clean the data: Remove rows with missing values
house_data = house_data.dropna(subset=features + [target])
print(f"Data cleaned. Remaining samples: {len(house_data)}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    house_data[features], house_data[target], test_size=0.2, random_state=0)
print("Data split into training and test sets.")

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
    'Linear Regression': LinearRegression()
}

# Evaluate models using cross-validation
cv_results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_results[name] = -scores.mean()  # Convert to positive MSE
    print(f"{name} cross-validation completed with MSE: {cv_results[name]}")

# Select the best model based on cross-validation MSE
best_model_name = min(cv_results, key=cv_results.get)
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', models[best_model_name])])
print(f"Best model based on cross-validation MSE: {best_model_name}")

# Train the best model on the full training set and evaluate it on the test set
best_pipeline.fit(X_train, y_train)
print("Best model trained on the full training set.")
y_pred = best_pipeline.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Print results
print(f"Final evaluation on test set - {best_model_name} with test MSE: {mse_test} and R^2: {r2_test}")