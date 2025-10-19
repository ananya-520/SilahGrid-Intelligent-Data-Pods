import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

NUM_SAMPLES = 1000
np.random.seed(42)

task_types = [
    'Preventive Maintenance','Disaster Recovery','Real-time Monitoring','Redundancy Validation',
    'AI Predictive Maintenance','Physical Access Control','Network Security','Data Encryption',
    'Vulnerability Management','Secure Data Disposal','Capacity Planning','Power/Cooling Optimization',
    'Automation','Asset Management','MLOps','Environmental Controls','Incident Management',
    'Workload Management','Automated Provisioning','Security Analytics'
]

def current_timestamp_iso(zone="Asia/Dubai"):
    """Return current time as ISO 8601 string in the requested timezone."""
    try:
        tz = ZoneInfo(zone)
    except Exception:
        tz = None
    if tz:
        return datetime.now(tz).isoformat()
    else:
        return datetime.utcnow().isoformat() + "Z"

data = []
for _ in range(NUM_SAMPLES):
    t = np.random.choice(task_types)
    cpu = float(np.random.uniform(10,95))
    battery = float(np.random.uniform(10,100))
    temperature = 20 + 0.3*cpu - 0.1*battery + np.random.normal(0,2)
    humidity = 30 + 0.1*cpu + 0.05*battery + np.random.normal(0,3)
    ozone_level = max(0.0, np.random.normal(0.01 + cpu*0.0003, 0.005))
    ts = current_timestamp_iso()  # timestamp per row
    data.append([ts, t, cpu, battery, temperature, humidity, ozone_level])

df = pd.DataFrame(data, columns=['timestamp','task_type','cpu_load','battery','temperature','humidity','ozone_level'])
csv_path = 'sensor_dataset.csv'
df.to_csv(csv_path, index=False)
print(f"[OK] {csv_path} created ({len(df)} rows)")
