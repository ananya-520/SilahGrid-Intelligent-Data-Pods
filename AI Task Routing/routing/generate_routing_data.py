# routing/generate_routing_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import joblib

# Config
OUT_CSV = "routing_dataset.csv"
NUM_SAMPLES = 500
NUM_PODS = 10
MAX_CAPACITY = 100.0
TEMP_SAFE_LIMIT = 75.0
BATTERY_SAFE_LIMIT = 10.0
MAX_LOCAL_TASK_SIZE = 5_000_000

SENSOR_MODEL_PATHS = ["../sensor_model.pkl", "sensor_model.pkl"]

# Load sensor model
sensor_model = None
for p in SENSOR_MODEL_PATHS:
    if os.path.exists(p):
        sensor_model = joblib.load(p)
        break
if sensor_model is None:
    raise FileNotFoundError("sensor_model.pkl not found.")

TASK_TYPES = [
    'Preventive Maintenance','Disaster Recovery','Real-time Monitoring','Redundancy Validation',
    'AI Predictive Maintenance','Physical Access Control','Network Security','Data Encryption',
    'Vulnerability Management','Secure Data Disposal','Capacity Planning','Power/Cooling Optimization',
    'Automation','Asset Management','MLOps','Environmental Controls','Incident Management',
    'Workload Management','Automated Provisioning','Security Analytics'
]

# Pod locations
PODS = [
    ("Dubai Internet City", 25.0950, 55.1595),
    ("Dubai Silicon Oasis", 25.1265, 55.3925),
    ("Dubai Marina", 25.0800, 55.1400),
    ("Jebel Ali Industrial Area", 24.9870, 55.0610),
    ("Business Bay", 25.1840, 55.2640),
    ("Al Quoz", 25.1360, 55.2200),
    ("Dubai Knowledge Park", 25.1020, 55.1630),
    ("Mirdif", 25.2190, 55.4160),
    ("Dubai Academic City", 25.1180, 55.4040),
    ("Deira", 25.2710, 55.3080)
]

def current_timestamp_iso(zone="Asia/Dubai"):
    try:
        tz = ZoneInfo(zone)
    except:
        tz = None
    return datetime.now(tz).isoformat() if tz else datetime.utcnow().isoformat() + "Z"

def sample_tasks(n_samples):
    return pd.DataFrame({
        "task_type": np.random.choice(TASK_TYPES, size=n_samples),
        "compute_cost": np.random.choice([5,10,15,20,30,40,60,80], size=n_samples),
        "data_size_bytes": np.random.choice([4000,200_000,500_000,2_000_000,10_000_000], size=n_samples)
    })

def sample_pods():
    return pd.DataFrame({
        "pod_name": [p[0] for p in PODS],
        "pod_lat": [p[1] for p in PODS],
        "pod_lon": [p[2] for p in PODS],
        "cpu_load": np.random.uniform(0, 90, size=NUM_PODS),
        "battery": np.random.uniform(5, 100, size=NUM_PODS),
        "max_capacity": MAX_CAPACITY
    })

def generate_dataset(n_samples=NUM_SAMPLES, out_csv=OUT_CSV):
    tasks = sample_tasks(n_samples)
    rows = []

    for i in range(n_samples):
        task = tasks.iloc[i:i+1]
        pods = sample_pods()

        # Vectorized sensor prediction
        task_rep = pd.concat([task]*NUM_PODS, ignore_index=True)
        df_pred = pd.concat([task_rep, pods[["cpu_load","battery"]].reset_index(drop=True)], axis=1)
        df_pred["hour"] = datetime.now().hour
        df_pred["minute"] = datetime.now().minute
        df_pred["day_of_week"] = datetime.now().weekday()

        preds = sensor_model.predict(df_pred)
        pods["temperature"] = preds[:,0]

        # Rule-based routing with dynamic pod selection
        chosen_idx = None
        for idx, pod in pods.iterrows():
            avail = pod["max_capacity"] - pod["cpu_load"]
            if (avail >= task["compute_cost"].values[0] and
                pod["temperature"] < TEMP_SAFE_LIMIT and
                pod["battery"] >= BATTERY_SAFE_LIMIT and
                task["data_size_bytes"].values[0] <= MAX_LOCAL_TASK_SIZE):
                chosen_idx = idx
                break

        if chosen_idx is None:
            label = NUM_PODS
            pod_cpu = pod_bat = pod_temp = -1
            chosen_name = ""
            chosen_lat = chosen_lon = np.nan
        else:
            label = chosen_idx
            pod_cpu = pods.loc[chosen_idx,"cpu_load"]
            pod_bat = pods.loc[chosen_idx,"battery"]
            pod_temp = pods.loc[chosen_idx,"temperature"]
            chosen_name = pods.loc[chosen_idx,"pod_name"]
            chosen_lat = pods.loc[chosen_idx,"pod_lat"]
            chosen_lon = pods.loc[chosen_idx,"pod_lon"]

        row = {
            "timestamp": current_timestamp_iso(),
            "task_type": task["task_type"].values[0],
            "compute_cost": task["compute_cost"].values[0],
            "data_size_bytes": task["data_size_bytes"].values[0],
            "pod_cpu_load": pod_cpu,
            "pod_battery": pod_bat,
            "pod_temperature": pod_temp,
            "num_pods": NUM_PODS,
            "avg_cpu_load": pods["cpu_load"].mean(),
            "label": int(label),
            "pod_name": chosen_name,
            "pod_lat": chosen_lat,
            "pod_lon": chosen_lon
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Routing dataset saved: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    generate_dataset()
