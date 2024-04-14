import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm

# Load the CSV data file
data_path = '../realtor-data_small.csv'  # Adjust path as needed
house_data = pd.read_csv(data_path)

# Imputing missing values for numerical columns
numerical_features = ['bed', 'bath', 'acre_lot', 'house_size']
numerical_transformer = SimpleImputer(strategy='median')

# One-hot encoding for categorical data
categorical_features = ['city']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessor for pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
model_lr = LinearRegression()

# Create the pipelines
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_rf)])
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_lr)])

# Extracting the feature columns and target column
X = house_data[numerical_features + categorical_features]
y = house_data['price']

# Remove rows with missing target (price)
X = X[y.notna()]
y = y[y.notna()]

# Setup KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform cross-validation
metrics_rf = {'MSE': [], 'MAE': [], 'R2': []}
metrics_lr = {'MSE': [], 'MAE': [], 'R2': []}

for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=kf.n_splits, desc="CV Folds")):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit and evaluate Random Forest
    pipeline_rf.fit(X_train, y_train)
    predictions_rf = pipeline_rf.predict(X_test)
    metrics_rf['MSE'].append(mean_squared_error(y_test, predictions_rf))
    metrics_rf['MAE'].append(mean_absolute_error(y_test, predictions_rf))
    metrics_rf['R2'].append(r2_score(y_test, predictions_rf))

    # Fit and evaluate Linear Regression
    pipeline_lr.fit(X_train, y_train)
    predictions_lr = pipeline_lr.predict(X_test)
    metrics_lr['MSE'].append(mean_squared_error(y_test, predictions_lr))
    metrics_lr['MAE'].append(mean_absolute_error(y_test, predictions_lr))
    metrics_lr['R2'].append(r2_score(y_test, predictions_lr))

# Print average metrics
print("Random Forest Average Metrics:", {k: sum(v) / len(v) for k, v in metrics_rf.items()})
print("Linear Regression Average Metrics:", {k: sum(v) / len(v) for k, v in metrics_lr.items()})
