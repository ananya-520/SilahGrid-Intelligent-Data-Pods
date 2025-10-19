# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------------------
# Load models
# ---------------------------
routing_model = joblib.load("routing/routing_model.pkl")
sensor_model = joblib.load("sensor_model.pkl")

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
# Predict pod temperature
# ---------------------------
def predict_pod_temperature(task_type, cpu, battery):
    df = pd.DataFrame([{
        "task_type": task_type,
        "cpu_load": cpu,
        "battery": battery,
        "hour": datetime.now().hour,
        "minute": datetime.now().minute,
        "day_of_week": datetime.now().weekday()
    }])
    return sensor_model.predict(df)[0][0]  # temperature

# ---------------------------
# Initialize pods in session state
# ---------------------------
if "pods_df" not in st.session_state:
    pods_df = pd.DataFrame({
        "pod_name": [p[0] for p in PODS],
        "cpu_load": np.random.uniform(0, 90, len(PODS)),
        "battery": np.random.uniform(10, 100, len(PODS)),
        "max_capacity": 100
    })
    pods_df["temperature"] = pods_df.apply(
        lambda row: predict_pod_temperature("Real-time Monitoring", row.cpu_load, row.battery), axis=1
    )
    pods_df["avail_capacity"] = pods_df["max_capacity"] - pods_df["cpu_load"]
    st.session_state.pods_df = pods_df
else:
    pods_df = st.session_state.pods_df

# ---------------------------
# Initialize tasks in session state
# ---------------------------
if "tasks_df" not in st.session_state:
    tasks_df = pd.DataFrame([
        {"task_type": "Real-time Monitoring", "compute_cost": 40, "data_size_bytes": 200000},
        {"task_type": "Network Security", "compute_cost": 30, "data_size_bytes": 50000},
        {"task_type": "AI Analytics", "compute_cost": 50, "data_size_bytes": 100000},
        {"task_type": "Capacity Planning", "compute_cost": 60, "data_size_bytes": 150000},
        {"task_type": "Impossible Task", "compute_cost": 200, "data_size_bytes": 500000},
        {"task_type": "Incident Management", "compute_cost": 35, "data_size_bytes": 70000},
        {"task_type": "Automated Provisioning", "compute_cost": 45, "data_size_bytes": 120000}
    ])
    st.session_state.tasks_df = tasks_df
else:
    tasks_df = st.session_state.tasks_df

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Task Routing Dashboard", layout="wide")
st.title("🚀 AI Task Routing Dashboard")

# Columns for metrics
col1, col2 = st.columns(2)
col1.metric("Total Tasks Pending", len(tasks_df))
col2.metric("Total Available Pods", len(pods_df))

st.subheader("Incoming Tasks")
st.dataframe(tasks_df)

st.subheader("Pods Status")
def color_avail(val):
    if val <= 0: return "background-color: #ff4d4d"  # red
    elif val < 30: return "background-color: #ffcc00"  # yellow
    else: return "background-color: #85e085"  # green

st.dataframe(pods_df.style.applymap(color_avail, subset=["avail_capacity"]))

# Task selection
task_idx = st.selectbox("Select a task to route:", tasks_df.index)
task = tasks_df.loc[task_idx]

# Route the task
if st.button("Route Task"):
    eligible = pods_df[
        (pods_df["avail_capacity"] >= task.compute_cost) &
        (pods_df["battery"] >= 10) &
        (pods_df["temperature"] < 75)
    ]

    if eligible.empty:
        chosen_pod = {"pod_name": "Central Datacenter"}
        reason = "No eligible pod. Routed to cloud."
    else:
        chosen_idx = eligible["avail_capacity"].idxmax()
        chosen_pod = pods_df.loc[chosen_idx]
        reason = "Pod with highest available capacity and safe temperature."
        # Update pod capacity
        pods_df.at[chosen_idx, "avail_capacity"] -= task.compute_cost
        st.session_state.pods_df = pods_df

    # Remove routed task
    st.session_state.tasks_df = tasks_df.drop(task_idx).reset_index(drop=True)

    st.subheader("Routing Decision")
    st.write(f"Task '{task.task_type}' routed to *{chosen_pod['pod_name']}*")
    st.info(f"Reason: {reason}")

    st.subheader("Updated Pods Status")
    st.dataframe(pods_df.style.applymap(color_avail, subset=["avail_capacity"]))

    st.subheader("Remaining Tasks")
    st.dataframe(st.session_state.tasks_df)
