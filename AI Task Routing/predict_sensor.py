# predict_sensor_with_timestamp.py
import joblib
import pandas as pd
from datetime import datetime

# Load trained model
model = joblib.load('sensor_model.pkl')

# Example new tasks
new_tasks = pd.DataFrame({
    'task_type': ['Real-time Monitoring', 'AI Predictive Maintenance'],
    'cpu_load': [55, 80],
    'battery': [70, 40],
    'timestamp': [datetime.now(), datetime.now()]  # current time for each task
})

# Extract timestamp features
new_tasks['hour'] = new_tasks['timestamp'].dt.hour
new_tasks['minute'] = new_tasks['timestamp'].dt.minute
new_tasks['day_of_week'] = new_tasks['timestamp'].dt.dayofweek

# Drop timestamp column (model only needs numeric features)
X_new = new_tasks[['task_type', 'cpu_load', 'battery', 'hour', 'minute', 'day_of_week']]

# Predict sensor values
predictions = model.predict(X_new)
pred_df = pd.DataFrame(predictions, columns=['temperature', 'humidity', 'ozone_level'])

# Combine input and predictions
result = pd.concat([X_new, pred_df], axis=1)
print(result)
