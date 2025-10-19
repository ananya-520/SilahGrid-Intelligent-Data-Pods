# smart_pods_simulation.py
import simpy
import random
import folium
import os
import sys
from pathlib import Path
from threejs_pages import save_threejs_page

# Add parent directory to path to import pod_config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pod_config import DUBAI_LOCATIONS, get_initial_value

# -----------------------------
# Pod Model
# -----------------------------
class SmartPod:
    """
    Represents an edge AI pod with sensors and power parameters.
    """
    def __init__(self, pod_id, name, latitude, longitude):
        self.pod_id = pod_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        # Use centralized initial values
        self.cpu_load = get_initial_value('cpu_load', pod_id) or random.uniform(10, 90)
        self.temperature = get_initial_value('temperature', pod_id) or random.uniform(25, 50)
        self.solar_power = get_initial_value('solar_power', pod_id) or random.uniform(200, 800)
        self.battery = get_initial_value('battery', pod_id) or random.uniform(40, 100)
        self.ozone_level = get_initial_value('ozone_level', pod_id) or random.uniform(0, 0.05)

    def update_status(self):
        """Simulates small random changes in pod status."""
        self.cpu_load = random.uniform(10, 90)
        self.temperature += random.uniform(-1, 1)
        self.solar_power = random.uniform(200, 800)
        self.battery = max(0, min(100,
            self.battery + self.solar_power * 0.001 - self.cpu_load * 0.005))
        self.ozone_level = random.uniform(0, 0.05)

    def get_status(self):
        return {
            "Pod ID": self.pod_id,
            "Name": self.name,
            "Location": (self.latitude, self.longitude),
            "CPU Load (%)": round(self.cpu_load, 2),
            "Temperature (°C)": round(self.temperature, 2),
            "Solar Power (W)": round(self.solar_power, 2),
            "Battery (%)": round(self.battery, 2),
            "Ozone (ppm)": round(self.ozone_level, 3)
        }

# -----------------------------
# Simulation
# -----------------------------
def pod_operation(env, pod):
    """Each pod updates its readings every simulated hour."""
    while True:
        pod.update_status()
        print(f"[Hour {env.now}] Updated Pod {pod.pod_id}: {pod.get_status()}")
        yield env.timeout(1)  # one simulation hour

def run_simulation(num_pods=10, duration=5):
    """Creates and runs a simulation with multiple SmartPods."""
    env = simpy.Environment()
    pods = []

    selected_locations = DUBAI_LOCATIONS[:num_pods]

    for i, (name, lat, lon) in enumerate(selected_locations):
        pod = SmartPod(i, name, lat, lon)
        pods.append(pod)
        env.process(pod_operation(env, pod))

    env.run(until=duration)
    return pods

# -----------------------------
# Folium Visualization
# -----------------------------

def create_pod_map(pods, output_file="smart_pods_map.html"):
    """Creates a Folium map showing pod locations, hover details, and clickable 3D page."""
    city_map = folium.Map(location=[25.0950, 55.1595], zoom_start=11)

    # Ensure folder for 3D pages
    os.makedirs("pod_pages", exist_ok=True)

    for pod in pods:
        status = pod.get_status()
        # Save a separate Three.js page for each pod
        save_threejs_page(pod)

        # Tooltip on hover with full pod details
        tooltip_html = f"""
<b>{pod.name} (ID: {pod.pod_id})</b><br>
CPU Load: {status['CPU Load (%)']}%<br>
Temperature: {status['Temperature (°C)']} °C<br>
Battery: {status['Battery (%)']}%<br>
Solar Power: {status['Solar Power (W)']} W<br>
Ozone Level: {status['Ozone (ppm)']} ppm
"""

        # Path to local 3D page
        local_page = f"pod_pages/pod_{pod.pod_id}.html"

        # Popup with a clickable link to open in new tab
        popup_html = f'<a href="{local_page}" target="_blank">Open 3D View</a>'

        # Add CircleMarker
        folium.CircleMarker(
            location=[pod.latitude, pod.longitude],
            radius=8,
            tooltip=tooltip_html,       # hover details
            popup=folium.Popup(popup_html, max_width=300),  # click opens page
            color='blue',
            fill=True,
            fill_color='cyan',
            fill_opacity=0.7
        ).add_to(city_map)

    city_map.save(output_file)
    abs_path = os.path.abspath(output_file)
    print(f"[OK] Folium map saved to: {abs_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("=== Smart Edge AI Pods Simulation ===")
    pods = run_simulation(num_pods=10, duration=3)
    create_pod_map(pods)
    print("=== Simulation & Map Complete ===")






