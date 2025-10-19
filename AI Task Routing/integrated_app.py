import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import pod_config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pod_config import DUBAI_LOCATIONS, POD_INITIAL_VALUES

# Suppress sklearn InconsistentVersionWarning when unpickling models saved with
# a slightly different scikit-learn patch version. We still catch load errors
# below and fallback to a deterministic heuristic if models fail to load.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Pod locations from centralized config
PODS = DUBAI_LOCATIONS


def _safe_load(path: Path):
    try:
        return joblib.load(str(path))
    except Exception:
        return None


def predict_pod_temperature(sensor_model, task_type, cpu, battery):
    if sensor_model is None:
        # fallback deterministic mapping
        return 20 + 0.3 * cpu - 0.1 * battery
    df = pd.DataFrame([{
        "task_type": task_type,
        "cpu_load": cpu,
        "battery": battery,
        "hour": datetime.now().hour,
        "minute": datetime.now().minute,
        "day_of_week": datetime.now().weekday()
    }])
    try:
        out = sensor_model.predict(df)
        # model may return array-like with shape (1,) or (1,1)
        if hasattr(out, '__len__'):
            return float(out[0][0]) if hasattr(out[0], '__len__') else float(out[0])
        return float(out)
    except Exception:
        return 25.0


def render_ai_routing():
    """Render the original AI Task Routing dashboard inside the main app.

    This function loads models from the `AI Task Routing/` folder when available and
    stores its UI state under distinct session keys so it won't interfere with the
    main simulator's `pods` objects.
    """
    base = Path.cwd() / 'AI Task Routing'
    routing_model = _safe_load(base / 'routing' / 'routing_model.pkl')
    sensor_model = _safe_load(base / 'sensor_model.pkl') or _safe_load(base / 'sensor_model.pkl')

    st.header("AI Task Routing Dashboard")

    # initialize pods_df in session state under a unique key
    if 'routing_pods_df' not in st.session_state:
        # Use centralized initial values for consistency
        pods_df = pd.DataFrame({
            'pod_name': [p[0] for p in PODS],
            'cpu_load': POD_INITIAL_VALUES['cpu_load'],
            'battery': POD_INITIAL_VALUES['battery'],
            'max_capacity': 100
        })
        pods_df['temperature'] = POD_INITIAL_VALUES['temperature']
        pods_df['avail_capacity'] = pods_df['max_capacity'] - pods_df['cpu_load']
        st.session_state.routing_pods_df = pods_df
    else:
        pods_df = st.session_state.routing_pods_df

    if 'routing_tasks_df' not in st.session_state:
        tasks_df = pd.DataFrame([
            {"task_type": "Real-time Monitoring", "compute_cost": 40, "data_size_bytes": 200000},
            {"task_type": "Network Security", "compute_cost": 30, "data_size_bytes": 50000},
            {"task_type": "AI Analytics", "compute_cost": 50, "data_size_bytes": 100000},
            {"task_type": "Capacity Planning", "compute_cost": 60, "data_size_bytes": 150000},
            {"task_type": "Impossible Task", "compute_cost": 200, "data_size_bytes": 500000},
            {"task_type": "Incident Management", "compute_cost": 35, "data_size_bytes": 70000},
            {"task_type": "Automated Provisioning", "compute_cost": 45, "data_size_bytes": 120000}
        ])
        st.session_state.routing_tasks_df = tasks_df
    else:
        tasks_df = st.session_state.routing_tasks_df

    col1, col2 = st.columns(2)
    col1.metric("Total Tasks Pending", len(tasks_df))
    col2.metric("Total Available Pods", len(pods_df))

    st.subheader("Incoming Tasks")
    st.dataframe(tasks_df)

    st.subheader("Pods Status")
    def _color_avail(val):
        if val <= 0: return "background-color: #ff4d4d"
        elif val < 30: return "background-color: #ffcc00"
        else: return "background-color: #85e085"

    # Prefer Styler.map (replacement for deprecated applymap). If the user's
    # pandas version doesn't support it, fallback to showing the plain frame.
    try:
        st.dataframe(pods_df.style.map(_color_avail, subset=["avail_capacity"]))
    except Exception:
        st.dataframe(pods_df)

    task_idx = st.selectbox("Select a task to route:", tasks_df.index)
    task = tasks_df.loc[task_idx]

    if st.button("Route Task"):
        eligible = pods_df[(pods_df['avail_capacity'] >= task.compute_cost) & (pods_df['battery'] >= 10) & (pods_df['temperature'] < 75)]
        if eligible.empty:
            chosen_pod = {"pod_name": "Central Datacenter"}
            reason = "No eligible pod. Routed to cloud."
        else:
            chosen_idx = eligible['avail_capacity'].idxmax()
            chosen_pod = pods_df.loc[chosen_idx]
            reason = "Pod with highest available capacity and safe temperature."
            pods_df.at[chosen_idx, 'avail_capacity'] -= task.compute_cost
            st.session_state.routing_pods_df = pods_df

        st.session_state.routing_tasks_df = tasks_df.drop(task_idx).reset_index(drop=True)

        st.subheader('Routing Decision')
        st.write(f"Task '{task.task_type}' routed to *{chosen_pod['pod_name']}*")
        st.info(f"Reason: {reason}")

        st.subheader('Updated Pods Status')
        try:
            st.dataframe(pods_df.style.map(_color_avail, subset=['avail_capacity']))
        except Exception:
            st.dataframe(pods_df)

        st.subheader('Remaining Tasks')
        st.dataframe(st.session_state.routing_tasks_df)
