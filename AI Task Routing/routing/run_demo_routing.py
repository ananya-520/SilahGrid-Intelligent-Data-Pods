# routing/run_demo_routing.py
"""
Demo: generate some pods, use sensor_model to predict their sensors, then call predict_route
"""

import random, joblib, os
from predict_routing import predict_route
import pandas as pd

# load sensor model (assume in parent or root)
SENSOR_MODEL_PATH = os.path.join("..","sensor_model.pkl")
if not os.path.exists("sensor_model.pkl") and os.path.exists(SENSOR_MODEL_PATH):
    # copy or reference: we'll load directly in the predict functions which expect sensors in pods,
    # but for demo we'll call the sensor model here to fill pod state dicts if available.
    sensor_model = joblib.load(SENSOR_MODEL_PATH)
else:
    sensor_model = None
    if os.path.exists("sensor_model.pkl"):
        sensor_model = joblib.load("sensor_model.pkl")

TASK_TYPES = ['Preventive Maintenance','Disaster Recovery','Real-time Monitoring','Redundancy Validation',
              'AI Predictive Maintenance','Physical Access Control','Network Security','Data Encryption',
              'Vulnerability Management','Secure Data Disposal','Capacity Planning','Power/Cooling Optimization',
              'Automation','Asset Management','MLOps','Environmental Controls','Incident Management',
              'Workload Management','Automated Provisioning','Security Analytics']

def predict_pod_sensors_local(task_type, cpu, battery):
    if sensor_model is None:
        # fallback deterministic mapping if sensor model missing
        temp = 20 + 0.3*cpu - 0.1*battery
        hum = 30 + 0.1*cpu + 0.05*battery
        ozone = max(0, 0.01 + cpu*0.0003)
        return float(temp), float(hum), float(ozone)
    df = pd.DataFrame([{"task_type": task_type, "cpu_load": cpu, "battery": battery}])
    out = sensor_model.predict(df)[0]
    return float(out[0]), float(out[1]), float(out[2])

def make_pods(num_pods=6):
    pods = []
    for i in range(num_pods):
        cpu = random.uniform(0, 85)
        battery = random.uniform(10, 100)
        temp, hum, ozone = predict_pod_sensors_local(random.choice(TASK_TYPES), cpu, battery)
        pods.append({
            "id": i,
            "cpu_load": cpu,
            "battery": battery,
            "temperature": temp,
            "humidity": hum,
            "ozone": ozone,
            "max_capacity": 100.0
        })
    return pods

def main():
    pods = make_pods(6)
    task = {"task_type": random.choice(TASK_TYPES), "compute_cost": random.choice([10,20,30,40]), "data_size_bytes": 500000}
    target, score = predict_route(task, pods)
    if target < len(pods):
        print(f"Routed to pod-{target} (score={score:.3f})")
    else:
        print(f"Routed to central datacenter (score={score:.3f})")
    print("Pods snapshot (first 3):")
    for p in pods[:3]:
        print(p)

if __name__ == "__main__":
    main()
