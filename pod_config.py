# pod_config.py
# Centralized pod configuration for SilahGrid
# All modules should import from this file to ensure consistency

import numpy as np

# Number of pods in the system
NUM_PODS = 10

# Pod locations in Dubai
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

# Initial realistic pod values (based on actual edge computing scenarios)
# These values represent pods at different operational states
POD_INITIAL_VALUES = {
    'cpu_load': [25.4, 68.2, 42.8, 55.1, 18.6, 82.5, 38.9, 71.3, 29.7, 64.8],  # % (5-95 range realistic)
    'temperature': [32.5, 48.3, 38.7, 42.1, 30.2, 52.8, 36.4, 49.6, 33.8, 45.9],  # °C (28-55 typical for edge pods)
    'solar_power': [450.0, 620.0, 380.0, 510.0, 290.0, 680.0, 410.0, 590.0, 340.0, 550.0],  # W (200-800 range)
    'battery': [78.5, 52.3, 85.2, 68.9, 92.1, 45.6, 81.3, 58.7, 88.4, 61.2],  # % (40-100 typical)
    'ozone_level': [0.022, 0.035, 0.018, 0.028, 0.015, 0.042, 0.020, 0.038, 0.016, 0.031]  # ppm (0-0.05 range)
}

# Pod operational scenarios (for simulation and plasma modulation)
POD_SCENARIOS = [
    'Light Load - Steady',
    'Heavy AI Training',
    'Burst Processing',
    'Medium Continuous',
    'Idle/Standby',
    'Critical Overload',
    'Variable Analytics',
    'Peak Performance',
    'Maintenance Mode',
    'Balanced Operations'
]

# Plasma modulation initial temperatures (for temperature-critical scenarios)
PLASMA_INITIAL_TEMPS = np.array([52.0, 88.0, 68.0, 78.0, 48.0, 90.0, 65.0, 92.0, 50.0, 72.0])

# Physical pod dimensions (meters)
POD_DIMENSIONS = {
    'length': 6.0,
    'width': 2.4,
    'height': 2.6
}

# Operating thresholds
THRESHOLDS = {
    'battery_max': 100.0,
    'battery_low': 20.0,
    'battery_critical': 10.0,
    'temp_normal': 25.0,
    'temp_threshold': 40.0,
    'temp_critical': 75.0,
    'solar_max': 30.0,  # Max solar input per tick for simulation
    'pod_capacity': 50  # Max task size for local processing
}

# Workload characteristics (for plasma modulation)
WORKLOAD_CHARACTERISTICS = {
    'base': np.array([0.6, 1.3, 0.9, 1.0, 0.4, 1.35, 0.85, 1.4, 0.5, 1.1]),
    'amplitude': np.array([0.1, 0.2, 0.5, 0.15, 0.05, 0.25, 0.4, 0.3, 0.08, 0.18]),
    'frequency': np.array([0.03, 0.05, 0.12, 0.04, 0.02, 0.06, 0.08, 0.07, 0.025, 0.055]),
    'phases': np.linspace(0, 2*np.pi, NUM_PODS)
}

def get_pod_location(pod_id):
    """Get location tuple for a specific pod (1-indexed)"""
    if 1 <= pod_id <= NUM_PODS:
        return DUBAI_LOCATIONS[pod_id - 1]
    return None

def get_initial_value(metric, pod_id):
    """Get initial value for a specific metric and pod (1-indexed)"""
    if 1 <= pod_id <= NUM_PODS and metric in POD_INITIAL_VALUES:
        return POD_INITIAL_VALUES[metric][pod_id - 1]
    return None

def get_pod_scenario(pod_id):
    """Get scenario description for a specific pod (1-indexed)"""
    if 1 <= pod_id <= NUM_PODS:
        return POD_SCENARIOS[pod_id - 1]
    return None
