# visualize_sensor_model.py
import matplotlib
matplotlib.use('Agg')  # non-GUI backend so the script works on headless machines

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Ensure you are in the same folder as sensor_dataset.csv and sensor_model.pkl ---
DATA_CSV = "sensor_dataset.csv"
MODEL_PKL = "sensor_model.pkl"

# Load dataset (must match the format used during training)
data = pd.read_csv(DATA_CSV)

# Check columns quickly
required_cols = {"task_type", "cpu_load", "battery", "temperature", "humidity", "ozone_level"}
missing = required_cols - set(data.columns)
if missing:
    raise RuntimeError(f"Missing columns in {DATA_CSV}: {missing}")

# Inputs (raw) and targets
X = data[["task_type", "cpu_load", "battery"]]   # <- raw task_type (string) expected by the pipeline
y = data[["temperature", "humidity", "ozone_level"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model (the Pipeline that includes the ColumnTransformer)
model = joblib.load(MODEL_PKL)

# Predict on test set (pass raw X_test so pipeline can do preprocessing)
y_pred = model.predict(X_test)

# plotting helper (saves PNG files)
def plot_pred_vs_actual(actual, predicted, sensor_name, filename):
    plt.figure(figsize=(6,4))
    plt.scatter(actual, predicted, color='blue', alpha=0.6)
    mn = min(actual.min(), predicted.min())
    mx = max(actual.max(), predicted.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{sensor_name} — Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Save three plots
plot_pred_vs_actual(y_test['temperature'], y_pred[:,0], "Temperature (°C)", "temperature_plot.png")
plot_pred_vs_actual(y_test['humidity'],    y_pred[:,1], "Humidity (%)",        "humidity_plot.png")
plot_pred_vs_actual(y_test['ozone_level'], y_pred[:,2], "Ozone Level (ppm)",   "ozone_plot.png")

print("[OK] Plots saved: temperature_plot.png, humidity_plot.png, ozone_plot.png")
