# routing/predict_routing_no_distance.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------
# Load models
# ---------------------------
model = joblib.load("routing_model.pkl")
sensor_model = joblib.load("../sensor_model.pkl")

# ---------------------------
# Pod locations
# ---------------------------
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

# ---------------------------
# Timestamp
# ---------------------------
def current_timestamp_iso(zone="Asia/Dubai"):
    try: tz = ZoneInfo(zone)
    except: tz=None
    return datetime.now(tz).isoformat() if tz else datetime.utcnow().isoformat() + "Z"

# ---------------------------
# Example incoming task
# ---------------------------
TASK = {
    "task_type":"Real-time Monitoring",
    "compute_cost":40,
    "data_size_bytes":200000,
    "lat":25.1180,
    "lon":55.4040
}

# ---------------------------
# Generate pod status
# ---------------------------
pods = pd.DataFrame({
    "pod_name":[p[0] for p in PODS],
    "pod_lat":[p[1] for p in PODS],
    "pod_lon":[p[2] for p in PODS],
    "cpu_load":np.random.uniform(0,90,len(PODS)),
    "battery":np.random.uniform(10,100,len(PODS)),
    "max_capacity":100
})

# ---------------------------
# Predict temperatures via sensor_model
# ---------------------------
task_rep = pd.concat([pd.DataFrame([TASK]*len(PODS))]*1, ignore_index=True)
df_pred = pd.concat([task_rep[["task_type","compute_cost","data_size_bytes"]].reset_index(drop=True), 
                     pods[["cpu_load","battery"]].reset_index(drop=True)], axis=1)
df_pred["hour"] = datetime.now().hour
df_pred["minute"] = datetime.now().minute
df_pred["day_of_week"] = datetime.now().weekday()

pods["temperature"] = sensor_model.predict(df_pred)[:,0]

# ---------------------------
# Eligible pods (without distance)
# ---------------------------
pods["avail_capacity"] = pods["max_capacity"] - pods["cpu_load"]
eligible = pods[(pods["avail_capacity"]>=TASK["compute_cost"]) &
                (pods["battery"]>=10) &
                (pods["temperature"]<75)]

if eligible.empty:
    chosen_pod = {"pod_name":"Central Datacenter", "pod_lat":np.nan, "pod_lon":np.nan}
    reason = "No eligible pod found. Routed to cloud datacenter."
else:
    # Pick the first eligible pod (distance ignored)
    chosen_pod = eligible.iloc[0]
    reason = f"First eligible pod with enough CPU and battery."

# ---------------------------
# Print routing decision
# ---------------------------
print("=== Routing Decision ===")
print(f"Time (Dubai): {current_timestamp_iso()}")
print(f"Task: type={TASK['task_type']}, compute_cost={TASK['compute_cost']}, data_size={TASK['data_size_bytes']}")
print(f"Decision: {chosen_pod['pod_name']}")
print(f"Reason: {reason}\n")

print("=== Per-pod status and scoring ===")
for _, pod in pods.iterrows():
    status = "ELIGIBLE" if pod["pod_name"] in eligible["pod_name"].values else "NOT-ELIGIBLE"
    print(f" pod-{pod.name}: name={pod['pod_name']}, avail_cap={pod['avail_capacity']:.1f}, "
          f"cpu={pod['cpu_load']:.1f}, bat={pod['battery']:.1f}, temp={pod['temperature']:.1f} => {status}")

# ---------------------------
# Append to dataset for retraining
# ---------------------------
new_row = {
    "timestamp": current_timestamp_iso(),
    "compute_cost": TASK["compute_cost"],
    "data_size_bytes": TASK["data_size_bytes"],
    "task_type": TASK["task_type"],
    "pod_cpu_load": chosen_pod.get("cpu_load", -1),
    "pod_battery": chosen_pod.get("battery", -1),
    "pod_temperature": chosen_pod.get("temperature", -1),
    "num_pods": len(PODS),
    "avg_cpu_load": pods["cpu_load"].mean(),
    "label": pods.index.get_loc(chosen_pod.name) if chosen_pod['pod_name'] != "Central Datacenter" else len(PODS),
    "pod_name": chosen_pod["pod_name"],
    "pod_lat": chosen_pod.get("pod_lat", np.nan),
    "pod_lon": chosen_pod.get("pod_lon", np.nan)
}
df_new = pd.DataFrame([new_row])
df_new.to_csv("routing_dataset.csv", mode="a", header=False, index=False)
print("\n[OK] New routing entry appended to dataset")
