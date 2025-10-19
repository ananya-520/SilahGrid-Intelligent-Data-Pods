# # pod_model.py
# import random

# class SmartPod:
#     """
#     Represents an edge AI pod with sensors and power parameters.
#     """
#     def __init__(self, pod_id, name, latitude, longitude):
#         self.pod_id = pod_id
#         self.name = name
#         self.latitude = latitude
#         self.longitude = longitude
#         self.cpu_load = random.uniform(10, 90)
#         self.temperature = random.uniform(25, 50)
#         self.solar_power = random.uniform(200, 800)
#         self.battery = random.uniform(40, 100)
#         self.ozone_level = random.uniform(0, 0.05)

#     def update_status(self):
#         """
#         Simulates small random changes in pod status.
#         """
#         self.cpu_load = random.uniform(10, 90)
#         self.temperature += random.uniform(-1, 1)
#         self.solar_power = random.uniform(200, 800)
#         self.battery = max(0, min(100,
#             self.battery + self.solar_power * 0.001 - self.cpu_load * 0.005))
#         self.ozone_level = random.uniform(0, 0.05)

#     def get_status(self):
#         return {
#             "Pod ID": self.pod_id,
#             "Name": self.name,
#             "Location": (self.latitude, self.longitude),
#             "CPU Load (%)": round(self.cpu_load, 2),
#             "Temperature (°C)": round(self.temperature, 2),
#             "Solar Power (W)": round(self.solar_power, 2),
#             "Battery (%)": round(self.battery, 2),
#             "Ozone (ppm)": round(self.ozone_level, 3)
#         }



import random
import time
import sys
from pathlib import Path

# Add parent directory to path to import pod_config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pod_config import get_initial_value

class SmartPod:
    """
    Represents a realistic Edge AI pod with sensors and power parameters.
    Simulates behavior over time: idle, charging, and AI processing tasks.
    """

    def __init__(self, pod_id, name, latitude, longitude):
        self.pod_id = pod_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        
        # Core system states - use centralized values
        self.cpu_load = get_initial_value('cpu_load', pod_id) or random.uniform(5, 25)
        self.temperature = get_initial_value('temperature', pod_id) or random.uniform(28, 35)
        self.solar_power = get_initial_value('solar_power', pod_id) or random.uniform(300, 700)
        self.battery = get_initial_value('battery', pod_id) or random.uniform(70, 100)
        self.ozone_level = get_initial_value('ozone_level', pod_id) or random.uniform(0.01, 0.04)
        self.status = "Idle"
        
        # AI task control
        self.task_timer = 0
        self.is_running_task = False

    def start_ai_task(self, duration=10):
        """Simulate starting an AI processing task."""
        if not self.is_running_task and self.battery > 15:
            self.is_running_task = True
            self.task_timer = duration
            self.status = "Running AI Task"
        elif self.battery <= 15:
            self.status = "Low Power Mode - Cannot Start Task"

    def update_status(self, sunlight_factor=1.0):
        """
        Updates pod readings based on real-time logic.
        - Battery drains slowly in idle, faster during tasks.
        - Temperature rises with CPU load.
        - Solar power charges battery if sunlight is present.
        """
        # Solar charging (varies with sunlight)
        self.solar_power = max(0, random.uniform(100, 800) * sunlight_factor)
        charge_rate = (self.solar_power / 1000) * 0.5  # charge per tick

        # AI task behavior
        if self.is_running_task:
            self.cpu_load = random.uniform(60, 95)
            self.temperature += random.uniform(0.3, 0.8)
            drain_rate = 0.7  # heavy drain during AI task
            self.task_timer -= 1

            if self.task_timer <= 0:
                self.is_running_task = False
                self.status = "Idle"

        else:
            # Idle mode
            self.cpu_load = random.uniform(5, 30)
            self.temperature += random.uniform(-0.2, 0.2)
            drain_rate = 0.05  # slow drain when idle

        # Battery calculation
        self.battery += charge_rate - drain_rate
        self.battery = max(0, min(100, self.battery))

        # Auto-status based on conditions
        if self.battery <= 10:
            self.status = "Low Power Mode"
            self.cpu_load = random.uniform(3, 10)
        elif self.battery < 30 and sunlight_factor > 0.5:
            self.status = "Charging"
        elif not self.is_running_task:
            self.status = "Idle"

        # Ozone reading fluctuates slightly
        self.ozone_level = max(0, random.uniform(0.01, 0.05))

    def get_status(self):
        """Return current pod metrics as dictionary."""
        return {
            "Pod ID": self.pod_id,
            "Name": self.name,
            "Status": self.status,
            "Location": (self.latitude, self.longitude),
            "CPU Load (%)": round(self.cpu_load, 2),
            "Temperature (°C)": round(self.temperature, 2),
            "Solar Power (W)": round(self.solar_power, 2),
            "Battery (%)": round(self.battery, 2),
            "Ozone (ppm)": round(self.ozone_level, 3)
        }


# Example usage
if __name__ == "__main__":
    pod = SmartPod(1, "Pod Alpha", 25.2, 55.3)
    for i in range(20):
        if i == 5:  # Start task at t=5
            pod.start_ai_task(duration=8)
        pod.update_status(sunlight_factor=random.uniform(0.6, 1.0))
        print(pod.get_status())
        time.sleep(1)
