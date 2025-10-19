# simulation_env.py
import simpy
import csv
import os
from pod_model import SmartPod

# Dubai pod locations with area names
DUBAI_LOCATIONS = [
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

def pod_operation(env, pod, writer):
    """
    Each pod updates its readings every simulated hour
    and logs the data for CSV/visualization.
    """
    while True:
        pod.update_status()
        status = pod.get_status()

        # Print live simulation output
        print(f"[Hour {env.now}] Updated Pod {pod.pod_id}: {status}")

        # Log data to CSV
        writer.writerow({
            "Time": env.now,
            "Pod_ID": pod.pod_id,
            "Name": pod.name,
            "Latitude": pod.latitude,
            "Longitude": pod.longitude,
            "CPU_Load": status["CPU Load (%)"],
            "Temperature": status["Temperature (°C)"],
            "Solar_Power": status["Solar Power (W)"],
            "Battery": status["Battery (%)"],
            "Ozone": status["Ozone (ppm)"]
        })

        yield env.timeout(1)  # one simulated hour


def run_simulation(num_pods=10, duration=5, output_csv="pod_data.csv"):
    """
    Creates and runs a simulation with multiple SmartPods.
    Exports the data to a CSV file for MATLAB or web visualization.
    """
    env = simpy.Environment()
    pods = []

    selected_locations = DUBAI_LOCATIONS[:num_pods]

    # Prepare CSV output
    with open(output_csv, mode="w", newline="") as file:
        fieldnames = [
            "Time", "Pod_ID", "Name", "Latitude", "Longitude",
            "CPU_Load", "Temperature", "Solar_Power",
            "Battery", "Ozone"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Create pods and register their processes
        for i, (name, lat, lon) in enumerate(selected_locations):
            pod = SmartPod(i, name, lat, lon)
            pods.append(pod)
            env.process(pod_operation(env, pod, writer))

        # Run the environment for the given duration
        env.run(until=duration)

    print(f"[OK] Simulation complete. Data saved to {os.path.abspath(output_csv)}")
    return pods
