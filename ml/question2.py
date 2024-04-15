import pandas as pd

# Load the dataset
file_path = '../realtor-data_small.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and the summary of the data
data.head(), data.info(), data.describe()

# Check the percentage of missing values in key columns
missing_data = data[['price', 'house_size']].isnull().mean() * 100

# We'll also check the correlation between house size and price to confirm our hypothesis
correlation = data[['house_size', 'price']].corr()

print(missing_data, correlation)

# Dropping rows where 'price' is missing
data_cleaned = data.dropna(subset=['price'])

# Imputing missing values in 'house_size' with the median
median_house_size = data_cleaned['house_size'].median()
data_cleaned['house_size'].fillna(median_house_size, inplace=True)

# Verify the imputation
print(data_cleaned[['price', 'house_size']].isnull().sum())

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data_cleaned[['house_size']]
y = data_cleaned['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Initialize the models
linear_model = LinearRegression()
ridge_model = Ridge(random_state=42)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Function to perform cross-validation and compute RMSE
def evaluate_model(model, X, y, cv=5):
    neg_rmse_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    rmse_scores = -neg_rmse_scores
    return rmse_scores.mean()

# Evaluate models
linear_rmse = evaluate_model(linear_model, X_train, y_train)
ridge_rmse = evaluate_model(ridge_model, X_train, y_train)
random_forest_rmse = evaluate_model(random_forest_model, X_train, y_train)

print(linear_rmse, ridge_rmse, random_forest_rmse)

# Fit the best model on the entire training data
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred = linear_model.predict(X_test)

# Calculate RMSE on the test set
from sklearn.metrics import mean_squared_error

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(test_rmse)