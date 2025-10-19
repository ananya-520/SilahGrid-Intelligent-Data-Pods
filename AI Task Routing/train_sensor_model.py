# train_sensor_model_with_timestamp.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv('sensor_dataset.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract features from timestamp
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Features and targets
X = df[['task_type', 'cpu_load', 'battery', 'hour', 'minute', 'day_of_week']]
y = df[['temperature', 'humidity', 'ozone_level']]

# One-hot encode task_type
preprocessor = ColumnTransformer(
    transformers=[('task_type', OneHotEncoder(), ['task_type'])],
    remainder='passthrough'
)

# Multi-output regression model
model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
])

# Train the model
model.fit(X, y)

# Save trained model
joblib.dump(model, 'sensor_model.pkl')
print("[OK] sensor_model.pkl trained with timestamp features and saved")
