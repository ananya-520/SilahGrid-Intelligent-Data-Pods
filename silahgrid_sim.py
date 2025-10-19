import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import deque
from pod_config import (
    NUM_PODS, DUBAI_LOCATIONS, POD_INITIAL_VALUES, 
    THRESHOLDS, get_initial_value, get_pod_location
)

# --- CONFIG ---
POD_CAPACITY = THRESHOLDS['pod_capacity']
BATTERY_MAX = THRESHOLDS['battery_max']
TEMP_THRESHOLD = THRESHOLDS['temp_threshold']
COOLING_RATE = 2.0  # Degrees per tick when cooling
SOLAR_MAX = THRESHOLDS['solar_max']
TASK_QUEUE_MAXLEN = 20

# --- POD CLASS ---
class Pod:
    def __init__(self, pod_id):
        self.pod_id = pod_id
        # Use centralized initial values
        self.battery = get_initial_value('battery', pod_id) or BATTERY_MAX * random.uniform(0.7, 1.0)
        self.temperature = get_initial_value('temperature', pod_id) or (25.0 + random.uniform(0, 5))
        self.cooling = False
        self.solar_input = get_initial_value('solar_power', pod_id) / 100.0 or 0.0  # Scaled for simulation
        self.load = get_initial_value('cpu_load', pod_id) / 10.0 or 0.0
        self.status = "Idle"
        self.alert = ""
        self.task_queue = deque(maxlen=TASK_QUEUE_MAXLEN)
        self.processed_tasks = 0
        self.offloaded_tasks = 0
        self.energy_used = 0.0
        self.energy_saved = 0.0
        self.last_decision = ""

    def update_solar(self, hour):
        # Simulate solar as a sine wave (max at noon)
        solar = max(0, np.sin((hour-6)/12 * np.pi)) * SOLAR_MAX
        self.solar_input = solar
        self.battery = min(BATTERY_MAX, self.battery + solar * 0.1)

    def add_task(self, task):
        self.task_queue.append(task)

    def process(self, data_center, tick):
        self.cooling = False
        self.alert = ""
        self.load = 0.0
        if self.task_queue:
            task = self.task_queue[0]
            # Decision logic
            if self.battery < 20.0:
                self.status = "Low Battery: Offloading"
                self.last_decision = "Offloaded (Low Battery)"
                data_center.add_task(task)
                self.task_queue.popleft()
                self.offloaded_tasks += 1
                self.alert = "Low battery! Task offloaded."
                return
            if self.temperature > TEMP_THRESHOLD:
                self.cooling = True
                self.status = "Cooling"
                self.last_decision = "Paused (Cooling)"
                self.alert = "Overheating! Cooling activated."
                self.temperature = max(25.0, self.temperature - COOLING_RATE)
                return
            if task['size'] <= POD_CAPACITY:
                # Process locally
                self.status = f"Processing Task {task['id']}"
                self.last_decision = "Local"
                self.load = task['size']
                # Simulate processing delay
                time.sleep(0.05)
                self.temperature += self.load * 0.05
                self.battery = max(0.0, self.battery - self.load * 0.05)
                self.energy_used += self.load * 0.05
                self.task_queue.popleft()
                self.processed_tasks += 1
            else:
                # Offload to data center
                self.status = f"Offloading Task {task['id']}"
                self.last_decision = "Offloaded (Large Task)"
                data_center.add_task(task)
                self.task_queue.popleft()
                self.offloaded_tasks += 1
                self.energy_saved += task['size'] * 0.05  # Simulated saving
        else:
            self.status = "Idle"
            self.last_decision = "Idle"
        # Passive cooling if not working
        if not self.cooling and self.temperature > 25.0:
            self.temperature -= 0.1
        self.temperature = max(20.0, self.temperature)

# --- DATA CENTER CLASS ---
class DataCenter:
    def __init__(self):
        self.task_queue = deque()
        self.processed_tasks = 0
        self.energy_used = 0.0

    def add_task(self, task):
        self.task_queue.append(task)

    def process(self):
        if self.task_queue:
            task = self.task_queue.popleft()
            # Simulate processing delay
            time.sleep(0.02)
            self.processed_tasks += 1
            self.energy_used += task['size'] * 0.2  # Data center is less efficient

# --- TASK GENERATION ---
def generate_task(task_id):
    return {
        'id': task_id,
        'size': random.randint(10, 100),
        'timestamp': time.time(),
        'anomaly': random.random() < 0.05  # 5% chance of anomaly
    }

# --- STREAMLIT DASHBOARD ---
st.set_page_config(page_title="SilahGrid Simulation", layout="wide")
st.title("SilahGrid: Decentralized Edge AI Network Simulator")

# --- SESSION STATE ---
if 'pods' not in st.session_state:
    st.session_state.pods = [Pod(i+1) for i in range(NUM_PODS)]
    st.session_state.data_center = DataCenter()
    st.session_state.task_id = 1
    st.session_state.history = []
    st.session_state.tick = 0
    st.session_state.anomaly_alert = ""

# If NUM_PODS changed while a session exists, resize the pods list so the map and
# dashboard reflect the new pod count without requiring a full session reset.
if 'pods' in st.session_state and len(st.session_state.pods) != NUM_PODS:
    current = len(st.session_state.pods)
    if current < NUM_PODS:
        for i in range(current, NUM_PODS):
            st.session_state.pods.append(Pod(i+1))
    else:
        st.session_state.pods = st.session_state.pods[:NUM_PODS]

# --- SIDEBAR ---
# Sidebar reserved for future use (left intentionally empty)
st.sidebar.header("Navigation")
# Navigation: Home or single-tab views
nav = st.sidebar.radio("Go to", ["Home", "Map + 3D Visualization", "AI Task Routing", "Plasma Modulation", "Plasma Visualization"], index=0, key='nav')

# --- AUTO RUN CONFIG ---
# Fixed auto-run behaviour: every tick runs automatically once per second.
AUTO_RUN = True
SPEED = 1  # ticks per second
TASKS_PER_TICK = 2
# Keep compatibility variables used by run_simulation
auto_tasks = True if AUTO_RUN else False
tasks_per_tick = TASKS_PER_TICK
manual_task = False

# Fixed simulation parameters (no UI controls)


# --- MAIN LOOP ---
def run_simulation():
    pods = st.session_state.pods
    data_center = st.session_state.data_center
    tick = st.session_state.tick
    hour = (tick // SPEED) % 24
    # Task generation
    new_tasks = []
    # always generate a fixed number of tasks per tick
    for _ in range(TASKS_PER_TICK):
        task = generate_task(st.session_state.task_id)
        st.session_state.task_id += 1
        pod = random.choice(pods)
        pod.add_task(task)
        new_tasks.append(task)
        if task['anomaly']:
            st.session_state.anomaly_alert = f"Anomaly detected in Task {task['id']}! Security alert."
    # Pod updates
    for pod in pods:
        pod.update_solar(hour)
        pod.process(data_center, tick)
    data_center.process()
    # Log history
    for i, pod in enumerate(pods):
        st.session_state.history.append({
            'tick': tick,
            'pod': pod.pod_id,
            'battery': pod.battery,
            'temp': pod.temperature,
            'solar': pod.solar_input,
            'load': pod.load,
            'status': pod.status,
            'decision': pod.last_decision,
            'alert': pod.alert,
            'processed': pod.processed_tasks,
            'offloaded': pod.offloaded_tasks
        })
    st.session_state.tick += 1

def safe_rerun():
    """Trigger a Streamlit rerun in a way that works across Streamlit versions.

    We prefer changing query params (which forces a rerun). If that API is
    unavailable, try experimental_rerun. If neither is available, return and
    let the page stay as-is.
    """
    # Preferred: set query params to force a rerun (modern API)
    try:
        st.query_params = {"_t": str(int(time.time()))}
        return
    except Exception:
        # If assigning query_params isn't available for some reason, fall back to toggling
        # a session flag which changes session state and triggers a re-execute.
        st.session_state['_rerun_flag'] = not st.session_state.get('_rerun_flag', False)
        return

# Auto-run loop: run every 1/SPEED seconds when AUTO_RUN is True
if AUTO_RUN:
    run_simulation()
    time.sleep(1.0 / SPEED)
    safe_rerun()
else:
    # allow manual single-step via a button if AUTO_RUN disabled
    run = st.button("Run Simulation Tick")
    if run:
        run_simulation()
        safe_rerun()
# --- DASHBOARD LAYOUT ---
# Note: pod selector and efficiency metrics are rendered only on the Home page

# Prepare dataframe and derived metrics (helpers moved out so we can reuse per-page)
df = pd.DataFrame(st.session_state.history)
def prepare_df(history):
    if not history:
        return pd.DataFrame()
    d = pd.DataFrame(history)
    # ensure numeric types
    for c in ['tick','pod','battery','temp','solar','load','processed','offloaded']:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.sort_values(['pod','tick'])
    # compute deltas for processed/offloaded
    if 'processed' in d:
        d['processed_delta'] = d.groupby('pod')['processed'].diff().fillna(d['processed'])
    else:
        d['processed_delta'] = 0
    if 'offloaded' in d:
        d['offloaded_delta'] = d.groupby('pod')['offloaded'].diff().fillna(d['offloaded'])
    else:
        d['offloaded_delta'] = 0
    return d

def ambient_temp_curve(ticks):
    # simple daily ambient: mean 25, amplitude 8
    return 25 + 8 * np.sin((np.array(ticks) / 24.0) * 2 * np.pi)

# prepared dataframe used by all rendering functions
d = prepare_df(st.session_state.history)

# Rendering helpers for individual tabs (allow full-width single-tab views)
def render_tab1_single(dframe, selected_pod_id):
    st.header('Single Pod (48 ticks)')
    if dframe.empty:
        st.info('No history yet')
        return
    if selected_pod_id is None:
        st.info("Select a single pod using the top 'Show data for:' dropdown to view per-pod metrics.")
        return
    max_tick = int(dframe['tick'].max()) if not dframe.empty else st.session_state.tick
    min_tick = max(0, max_tick - 48)
    sel = dframe[(dframe['pod'] == selected_pod_id) & (dframe['tick'] >= min_tick)]
    fell_back = False
    if sel.empty:
        sel_all = dframe[dframe['pod'] == selected_pod_id]
        if not sel_all.empty:
            fell_back = True
            sel = sel_all
    if sel.empty:
        st.info('No data for selected pod yet')
        return
    if fell_back:
        st.info('No data in the recent 48-tick window; showing all available history for this pod.')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel['tick'], y=sel['battery'], name='Battery', mode='lines', line=dict(color='#1f77b4'), showlegend=False))
    fig.add_trace(go.Scatter(x=sel['tick'], y=sel['temp'], name='Temp (C)', mode='lines', line=dict(color='#ff7f0e'), yaxis='y2', showlegend=False))
    fig.add_trace(go.Bar(x=sel['tick'], y=sel['processed_delta'], name='Processed per tick', opacity=0.6, marker_color='#2ca02c', showlegend=False))
    fig.update_layout(title=f'Pod {selected_pod_id} performance', xaxis_title='Tick', yaxis_title='Battery / Tasks', yaxis2=dict(title='Temp (C)', overlaying='y', side='right'), uirevision=f"pod-{selected_pod_id}", margin=dict(b=80, l=60), height=420)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    # legend
    legend_cols = st.columns([1,1,1])
    legend_cols[0].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#1f77b4;margin-right:8px;border-radius:2px'></div>Battery</div>", unsafe_allow_html=True)
    legend_cols[1].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#ff7f0e;margin-right:8px;border-radius:2px'></div>Temp (C)</div>", unsafe_allow_html=True)
    legend_cols[2].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#2ca02c;margin-right:8px;border-radius:2px'></div>Processed per tick</div>", unsafe_allow_html=True)
    ind1, ind2, ind3 = st.columns([1,1,1])
    last = sel.iloc[-1]
    ind1.metric('Battery', f"{last['battery']:.1f}%")
    ind2.metric('Temp (C)', f"{last['temp']:.1f}")
    ind3.metric('Processed (last)', f"{int(last.get('processed_delta',0))}")

def render_tab2_single(dframe):
    st.header('Local vs Cloud')
    if dframe.empty:
        st.info('No history yet')
        return
    agg = dframe.groupby('tick').agg({'processed_delta':'sum','offloaded_delta':'sum'}).reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=agg['tick'], y=agg['processed_delta'], name='Processed (Local)', stackgroup='one', line=dict(color='#2ca02c')))
    fig2.add_trace(go.Scatter(x=agg['tick'], y=agg['offloaded_delta'], name='Offloaded (Cloud)', stackgroup='one', line=dict(color='#d62728')))
    fig2.update_layout(title='Local vs Offloaded Tasks Over Time', xaxis_title='Tick', yaxis_title='Tasks', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    a1, a2, a3 = st.columns([1,1,1])
    latest = agg.iloc[-1] if not agg.empty else None
    if latest is not None:
        a1.metric('Processed (latest tick)', int(latest['processed_delta']))
        a2.metric('Offloaded (latest tick)', int(latest['offloaded_delta']))
        a3.metric('Total Tasks (so far)', int(agg['processed_delta'].sum() + agg['offloaded_delta'].sum()))
    else:
        a1.metric('Processed (latest tick)', 'N/A')
        a2.metric('Offloaded (latest tick)', 'N/A')
        a3.metric('Total Tasks (so far)', 'N/A')

def render_tab3_single(dframe):
    st.header('Internal vs Ambient Temperature')
    if dframe.empty:
        st.info('No history yet')
        return
    avg_temp = dframe.groupby('tick')['temp'].mean().reset_index()
    ticks = avg_temp['tick'].values
    ambient = ambient_temp_curve(ticks)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=avg_temp['tick'], y=avg_temp['temp'], name='Internal Avg Temp', line=dict(color='#1f77b4')))
    fig3.add_trace(go.Scatter(x=avg_temp['tick'], y=ambient, name='Ambient Temp', line=dict(dash='dash', color='#7f7f7f')))
    fig3.update_layout(title='Internal vs Ambient Temperature', xaxis_title='Tick', yaxis_title='Temp (C)', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    b1, b2 = st.columns([1,1])
    if not avg_temp.empty:
        last_idx = avg_temp.iloc[-1]
        last_tick = int(last_idx['tick'])
        last_internal = float(last_idx['temp'])
        last_ambient = float(ambient[-1])
        b1.metric('Internal Avg Temp', f"{last_internal:.1f}°C")
        b2.metric('Ambient Temp', f"{last_ambient:.1f}°C")
    else:
        b1.metric('Internal Avg Temp', 'N/A')
        b2.metric('Ambient Temp', 'N/A')

# make the right-most column (graphs) slightly wider without taking the whole page
if nav == 'Home':
    # Pod selector (choose one or All) - shown only on Home
    pod_ids = [p.pod_id for p in st.session_state.pods]
    pod_options = ["All"] + [f"Pod {i}" for i in pod_ids]
    selected_pod_label = st.selectbox("Show data for:", pod_options, index=0)
    selected_pod_id = None
    if selected_pod_label != "All":
        try:
            selected_pod_id = int(selected_pod_label.split()[1])
        except Exception:
            selected_pod_id = None

    col1, col2, col3 = st.columns([2,2,3])

    with col1:
        st.subheader("Pod Metrics")
        for pod in st.session_state.pods:
            if selected_pod_id is not None and pod.pod_id != selected_pod_id:
                continue
            st.markdown(f"**Pod {pod.pod_id}**")
            st.progress(pod.battery / BATTERY_MAX, text=f"Battery: {pod.battery:.1f}%")
            st.progress(min(pod.temperature/60,1.0), text=f"Temp: {pod.temperature:.1f}°C")
            st.progress(min(pod.solar_input/SOLAR_MAX,1.0), text=f"Solar: {pod.solar_input:.1f}")
            st.text(f"Status: {pod.status}")
            st.text(f"Decision: {pod.last_decision}")
            if pod.alert:
                st.error(pod.alert)
            st.divider()

    with col2:
        st.subheader("Workload & Routing")
        for pod in st.session_state.pods:
            if selected_pod_id is not None and pod.pod_id != selected_pod_id:
                continue
            st.text(f"Pod {pod.pod_id} Queue: {len(pod.task_queue)} tasks")
            st.text(f"Processed: {pod.processed_tasks} | Offloaded: {pod.offloaded_tasks}")
        st.text(f"Data Center Queue: {len(st.session_state.data_center.task_queue)}")
        st.text(f"Data Center Processed: {st.session_state.data_center.processed_tasks}")
        st.divider()
        if st.session_state.anomaly_alert:
            st.warning(st.session_state.anomaly_alert)
            st.session_state.anomaly_alert = ""

    with col3:
        st.subheader("Live Graphs")
        # Home view uses built-in tabs to show the original graphs
        tabs = st.tabs(['Single Pod', 'AI Task Routing', 'Plasma Modulation', 'Pod Load Balance'])

        # Tab 1: Single Pod Performance (Map + 3D Visualization placeholder uses the Single Pod graph for now)
        with tabs[0]:
            if d.empty:
                st.info('No history yet')
            else:
                pod_sel = selected_pod_id
                max_tick = int(d['tick'].max()) if not d.empty else st.session_state.tick
                min_tick = max(0, max_tick - 48)
                if pod_sel is None:
                    st.info("Select a single pod using the top 'Show data for:' dropdown to view per-pod metrics.")
                else:
                    sel = d[(d['pod'] == pod_sel) & (d['tick'] >= min_tick)] if not d.empty else pd.DataFrame()
                    fell_back = False
                    if sel.empty and not d.empty:
                        sel_all = d[d['pod'] == pod_sel]
                        if not sel_all.empty:
                            fell_back = True
                            sel = sel_all
                    if sel.empty:
                        st.info('No data for selected pod yet')
                        st.empty()
                    else:
                        if fell_back:
                            st.info('No data in the recent 48-tick window; showing all available history for this pod.')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=sel['tick'], y=sel['battery'], name='Battery', mode='lines', line=dict(color='#1f77b4'), showlegend=False))
                        fig.add_trace(go.Scatter(x=sel['tick'], y=sel['temp'], name='Temp (C)', mode='lines', line=dict(color='#ff7f0e'), yaxis='y2', showlegend=False))
                        fig.add_trace(go.Bar(x=sel['tick'], y=sel['processed_delta'], name='Processed per tick', opacity=0.6, marker_color='#2ca02c', showlegend=False))
                        fig.update_layout(title=f'Pod {pod_sel} performance', xaxis_title='Tick', yaxis_title='Battery / Tasks', yaxis2=dict(title='Temp (C)', overlaying='y', side='right'), uirevision=f"pod-{pod_sel}", margin=dict(b=80, l=60), height=420)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        legend_cols = st.columns([1,1,1])
                        legend_cols[0].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#1f77b4;margin-right:8px;border-radius:2px'></div>Battery</div>", unsafe_allow_html=True)
                        legend_cols[1].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#ff7f0e;margin-right:8px;border-radius:2px'></div>Temp (C)</div>", unsafe_allow_html=True)
                        legend_cols[2].markdown("<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:14px;height:10px;background:#2ca02c;margin-right:8px;border-radius:2px'></div>Processed per tick</div>", unsafe_allow_html=True)
                        ind1, ind2, ind3 = st.columns([1,1,1])
                        if not sel.empty:
                            last = sel.iloc[-1]
                            ind1.metric('Battery', f"{last['battery']:.1f}%")
                            ind2.metric('Temp (C)', f"{last['temp']:.1f}")
                            ind3.metric('Processed (last)', f"{int(last.get('processed_delta',0))}")
                        else:
                            ind1.metric('Battery', 'N/A')
                            ind2.metric('Temp (C)', 'N/A')
                            ind3.metric('Processed (last)', 'N/A')

        # Tab 2: Local vs Cloud Tasks (AI Task Routing placeholder uses the Local vs Cloud graph for now)
        with tabs[1]:
            if d.empty:
                st.info('No history yet')
            else:
                agg = d.groupby('tick').agg({'processed_delta':'sum','offloaded_delta':'sum'}).reset_index()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=agg['tick'], y=agg['processed_delta'], name='Processed (Local)', stackgroup='one', line=dict(color='#2ca02c')))
                fig2.add_trace(go.Scatter(x=agg['tick'], y=agg['offloaded_delta'], name='Offloaded (Cloud)', stackgroup='one', line=dict(color='#d62728')))
                fig2.update_layout(title='Local vs Offloaded Tasks Over Time', xaxis_title='Tick', yaxis_title='Tasks', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
                st.markdown('---')
                a1, a2, a3 = st.columns([1,1,1])
                latest = agg.iloc[-1] if not agg.empty else None
                if latest is not None:
                    a1.metric('Processed (latest tick)', int(latest['processed_delta']))
                    a2.metric('Offloaded (latest tick)', int(latest['offloaded_delta']))
                    a3.metric('Total Tasks (so far)', int(agg['processed_delta'].sum() + agg['offloaded_delta'].sum()))
                else:
                    a1.metric('Processed (latest tick)', 'N/A')
                    a2.metric('Offloaded (latest tick)', 'N/A')
                    a3.metric('Total Tasks (so far)', 'N/A')

        # Tab 3: Internal vs Ambient Temperature (Plasma Madulation placeholder uses the same graph)
        with tabs[2]:
            if d.empty:
                st.info('No history yet')
            else:
                avg_temp = d.groupby('tick')['temp'].mean().reset_index()
                ticks = avg_temp['tick'].values
                ambient = ambient_temp_curve(ticks)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=avg_temp['tick'], y=avg_temp['temp'], name='Internal Avg Temp', line=dict(color='#1f77b4')))
                fig3.add_trace(go.Scatter(x=avg_temp['tick'], y=ambient, name='Ambient Temp', line=dict(dash='dash', color='#7f7f7f')))
                fig3.update_layout(title='Internal vs Ambient Temperature', xaxis_title='Tick', yaxis_title='Temp (C)', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
                st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
                st.markdown('---')
                b1, b2 = st.columns([1,1])
                if not avg_temp.empty:
                    last_idx = avg_temp.iloc[-1]
                    last_tick = int(last_idx['tick'])
                    last_internal = float(last_idx['temp'])
                    last_ambient = float(ambient[-1])
                    b1.metric('Internal Avg Temp', f"{last_internal:.1f}°C")
                    b2.metric('Ambient Temp', f"{last_ambient:.1f}°C")
                else:
                    b1.metric('Internal Avg Temp', 'N/A')
                    b2.metric('Ambient Temp', 'N/A')

        # Tab 4: Pod Load Balance
        with tabs[3]:
            if d.empty:
                st.info('No history yet')
            else:
                load_df = d.groupby(['tick','pod'])['load'].sum().reset_index()
                latest_tick = int(load_df['tick'].max())
                latest = load_df[load_df['tick'] == latest_tick]
                fig4 = px.bar(latest, x='pod', y='load', title=f'Pod load distribution at tick {latest_tick}')
                fig4.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
                st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
                stacked = load_df.pivot(index='tick', columns='pod', values='load').fillna(0)
                fig5 = go.Figure()
                for col in stacked.columns:
                    fig5.add_trace(go.Scatter(x=stacked.index, y=stacked[col], name=f'Pod {col}', stackgroup='one'))
                fig5.update_layout(title='Load per pod over time (stacked)', xaxis_title='Tick', yaxis_title='Load', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=12)), margin=dict(b=80, l=60), height=420)
                st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
                st.markdown('---')
                c1, c2, c3 = st.columns([1,1,1])
                total_load = load_df['load'].sum()
                c1.metric('Total Load (sum)', f"{total_load:.1f}")
                c2.metric('Max Load (per pod)', f"{load_df.groupby('pod')['load'].sum().max():.1f}")
                c3.metric('Pods', f"{len(st.session_state.pods)}")

else:
    # Sidebar-selected full-page tabs (placeholders). These should be empty so you can
    # drop unrelated content here later.
    if nav == 'Map + 3D Visualization':
        st.header('Map + 3D Visualization')
        st.info('Full-page map view (live). This uses the Map + 3D Visualization SmartPod model.')

        # Live integration (Option B): import SmartPod from the Map folder and maintain
        # a separate set of map pods in session_state. Update them each tick and regenerate
        # the folium map for embedding.
        try:
            from pathlib import Path
            import importlib.util
            import streamlit.components.v1 as components
            import sys

            pod_model_path = Path.cwd() / 'Map + 3D Visualization' / 'pod_model.py'
            vis_path = Path.cwd() / 'Map + 3D Visualization' / 'visualization.py'

            # Ensure the Map folder is on sys.path so local imports (threejs_pages, etc.) resolve
            map_dir = str(Path.cwd() / 'Map + 3D Visualization')
            if map_dir not in sys.path:
                sys.path.insert(0, map_dir)

            # Load SmartPod class dynamically
            SmartPod = None
            if pod_model_path.exists():
                spec_pm = importlib.util.spec_from_file_location('map_pod_model', str(pod_model_path))
                pm = importlib.util.module_from_spec(spec_pm)
                spec_pm.loader.exec_module(pm)
                SmartPod = getattr(pm, 'SmartPod', None)

            # Ensure map pods exist in session state
            if 'map_pods' not in st.session_state or len(st.session_state.get('map_pods', [])) != len(st.session_state.pods):
                st.session_state.map_pods = []

                # Try to load visualization locations to place pods correctly
                dubai_locations = None
                try:
                    if vis_path.exists():
                        spec_vis2 = importlib.util.spec_from_file_location('map_visualization2', str(vis_path))
                        vis2 = importlib.util.module_from_spec(spec_vis2)
                        spec_vis2.loader.exec_module(vis2)
                        dubai_locations = getattr(vis2, 'DUBAI_LOCATIONS', None)
                except Exception:
                    dubai_locations = None

                if SmartPod is not None:
                    # create SmartPod adapters that mirror the main pods' IDs
                    for idx, p in enumerate(st.session_state.pods):
                        # Prefer coordinates from DUBAI_LOCATIONS if available
                        if dubai_locations and idx < len(dubai_locations):
                            name, lat, lon = dubai_locations[idx]
                        else:
                            name = f"Pod {p.pod_id}"
                            lat = 25.0950 + (p.pod_id - 1) * 0.01
                            lon = 55.1595 + (p.pod_id - 1) * 0.01
                        mp = SmartPod(p.pod_id, name, lat, lon)
                        # initialize battery/temp to match existing pod
                        mp.battery = p.battery
                        mp.temperature = p.temperature
                        st.session_state.map_pods.append(mp)
                else:
                    # Fallback: create lightweight adapters
                    class _FallbackPod:
                        def __init__(self, pod, lat, lon, name=None):
                            self.pod_id = pod.pod_id
                            self.name = name or f"Pod {pod.pod_id}"
                            self.latitude = lat
                            self.longitude = lon
                            self.battery = pod.battery
                            self.temperature = pod.temperature
                        def update_status(self, *a, **k):
                            return
                        def get_status(self):
                            return {
                                'CPU Load (%)': 0,
                                'Temperature (°C)': round(self.temperature,2),
                                'Solar Power (W)': 0,
                                'Battery (%)': round(self.battery,2),
                                'Ozone (ppm)': 0.02
                            }
                    # build fallback pods using dubai_locations if available
                    for idx, p in enumerate(st.session_state.pods):
                        if dubai_locations and idx < len(dubai_locations):
                            _, lat, lon = dubai_locations[idx]
                        else:
                            lat = 25.0950 + (p.pod_id - 1) * 0.01
                            lon = 55.1595 + (p.pod_id - 1) * 0.01
                        st.session_state.map_pods.append(_FallbackPod(p, lat, lon))

            # Update map pods each tick: call update_status with a sunlight factor
            sunlight = max(0.1, min(1.0, np.sin((st.session_state.tick % 24 - 6) / 12 * np.pi) * 1.0))
            for mp in st.session_state.map_pods:
                try:
                    mp.update_status(sunlight_factor=sunlight)
                except TypeError:
                    # some implementations use no args
                    try:
                        mp.update_status()
                    except Exception:
                        pass

            # Create the folium map via visualization.create_pod_map if available
            if vis_path.exists():
                spec_vis = importlib.util.spec_from_file_location('map_visualization', str(vis_path))
                vis = importlib.util.module_from_spec(spec_vis)
                spec_vis.loader.exec_module(vis)
                out_file = 'smart_pods_map_live.html'
                try:
                    vis.create_pod_map(st.session_state.map_pods, output_file=out_file)
                    html = Path(out_file).read_text(encoding='utf-8')
                    components.html(html, height=750, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to generate embedded live map: {e}")
            else:
                st.warning('Visualization helper not found; cannot render map.')

            # Below the map: allow embedding a per-pod 3D page (generated by threejs_pages.save_threejs_page)
            try:
                pod_pages_dir = Path.cwd() / 'pod_pages'
                pod_options = ['None']
                if pod_pages_dir.exists():
                    # list available pod HTML pages and present them as choices
                    pages = sorted(pod_pages_dir.glob('pod_*.html'))
                    for p in pages:
                        # extract id from filename
                        name = p.stem  # pod_1
                        try:
                            pid = int(name.split('_')[1])
                        except Exception:
                            pid = None
                        if pid is not None:
                            pod_options.append(f'Pod {pid}')

                selected_3d = st.selectbox('Embed pod 3D view (select Pod)', pod_options, index=0)
                if selected_3d != 'None':
                    pid = int(selected_3d.split()[1])
                    page_file = pod_pages_dir / f'pod_{pid}.html'
                    if page_file.exists():
                        page_html = page_file.read_text(encoding='utf-8')
                        components.html(page_html, height=700, scrolling=True)
                    else:
                        st.warning(f'3D page for Pod {pid} not found (expected {page_file})')
            except Exception as e:
                st.error(f'Failed to embed 3D pod page: {e}')

        except Exception as e:
            st.error(f"Live map integration failed: {e}")

    elif nav == 'AI Task Routing':
        # Call the integrated original AI Task Routing app which lives in the
        # `AI Task Routing/` folder. This preserves the original models and UI.
        try:
            from AI_Task_Routing.integrated_app import render_ai_routing
        except Exception:
            # try with the folder name containing a space
            from importlib import util
            from pathlib import Path
            p = Path.cwd() / 'AI Task Routing' / 'integrated_app.py'
            spec = util.spec_from_file_location('ai_routing_integrated', str(p))
            mod = util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            render_ai_routing = getattr(mod, 'render_ai_routing')
        try:
            render_ai_routing()
        except Exception as e:
            st.error(f'Failed to render integrated AI Task Routing app: {e}')

    elif nav == 'Plasma Modulation':
        # Integrated Plasma Modulation view
        from scipy.ndimage import laplace
        from pod_config import POD_SCENARIOS, PLASMA_INITIAL_TEMPS, POD_DIMENSIONS, WORKLOAD_CHARACTERISTICS
        
        st.header('Plasma Cooling Network Metrics')
        
        # Initialize plasma simulation parameters in session state
        if 'plasma_initialized' not in st.session_state:
            st.session_state.plasma_initialized = False
            st.session_state.plasma_results = None
        
        # Sidebar configuration
        with st.sidebar:
            st.subheader("Plasma Configuration")
            num_steps = st.slider("Simulation Steps", 20, 100, 70, 10, key='plasma_steps')
            dt = st.slider("Time Step (s)", 1.0, 10.0, 4.2, 0.5, key='plasma_dt')
            grid_res = st.slider("Grid Resolution", 10, 20, 15, 1, key='plasma_grid')
            
            st.subheader("Temperature Thresholds (°C)")
            T_ambient = st.number_input("Ambient Temperature", 20, 50, 45, key='plasma_ambient')
            T_plasma_on = st.number_input("Plasma Activation", 40, 70, 55, key='plasma_on')
            T_plasma_off = st.number_input("Plasma Deactivation", 35, 60, 45, key='plasma_off')
            T_critical = st.number_input("Critical Temperature", 60, 90, 75, key='plasma_critical')
        
        # Pod configuration from centralized config
        pod_scenarios = POD_SCENARIOS
        max_temps_initial = PLASMA_INITIAL_TEMPS
        workload_base = WORKLOAD_CHARACTERISTICS['base']
        workload_amplitude = WORKLOAD_CHARACTERISTICS['amplitude']
        workload_frequency = WORKLOAD_CHARACTERISTICS['frequency']
        workload_phases = WORKLOAD_CHARACTERISTICS['phases']
        
        # Physical constants
        k_air = 0.028
        rho_air = 1.2
        cp_air = 1005
        alpha = k_air / (rho_air * cp_air)
        
        # Pod dimensions
        pod_length = POD_DIMENSIONS['length']
        pod_width = POD_DIMENSIONS['width']
        pod_height = POD_DIMENSIONS['height']
        
        def initialize_plasma_simulation():
            """Initialize the plasma simulation state"""
            X, Y, Z = np.meshgrid(
                np.linspace(0, pod_length, grid_res),
                np.linspace(0, pod_width, grid_res),
                np.linspace(0, pod_height, grid_res)
            )
            
            processor_zones = np.zeros(X.shape, dtype=bool)
            processor_zones[5:10, 7:12, 9:14] = True
            processor_zones[11:14, 7:12, 7:11] = True
            
            T_pods = np.ones((grid_res, grid_res, grid_res, NUM_PODS)) * T_ambient
            
            for i in range(NUM_PODS):
                T = T_pods[:, :, :, i].copy()
                T[processor_zones] = max_temps_initial[i]
                
                for z in range(grid_res):
                    heat_factor = (z / grid_res) ** 2
                    T[:, :, z] += heat_factor * (max_temps_initial[i] - T_ambient) * 0.3
                
                T_pods[:, :, :, i] = T
            
            num_electrodes = 8
            electrode_x = np.linspace(0.8, pod_length - 0.8, num_electrodes)
            electrode_y = np.ones(num_electrodes) * pod_width / 2
            electrode_z = np.ones(num_electrodes) * pod_height * 0.9
            E_field = 6000
            
            u_base = np.zeros(X.shape)
            v_base = np.zeros(X.shape)
            w_base = np.zeros(X.shape)
            
            for i in range(num_electrodes):
                dist = np.sqrt((X - electrode_x[i])**2 + (Y - electrode_y[i])**2 + (Z - electrode_z[i])**2)
                E_local = E_field / (dist**2 + 0.05)
                u_base += 1.0 * E_local * (X - electrode_x[i]) / (dist + 0.05)
                v_base += 1.0 * E_local * (Y - electrode_y[i]) / (dist + 0.05)
                w_base -= 4.0 * E_local
            
            velocity_magnitude = np.sqrt(u_base**2 + v_base**2 + w_base**2)
            max_velocity = 5.5
            u_base = u_base / (velocity_magnitude + 0.01) * max_velocity
            v_base = v_base / (velocity_magnitude + 0.01) * max_velocity
            w_base = w_base / (velocity_magnitude + 0.01) * max_velocity
            
            return T_pods, processor_zones, u_base, v_base, w_base, X, Y, Z
        
        def run_plasma_simulation(T_pods, processor_zones, u_base, v_base, w_base):
            """Run the plasma cooling simulation"""
            dx = pod_length / (grid_res - 1)
            
            max_temps = np.zeros((NUM_PODS, num_steps))
            avg_temps = np.zeros((NUM_PODS, num_steps))
            plasma_status = np.zeros((NUM_PODS, num_steps))
            power_consumption = np.zeros((NUM_PODS, num_steps))
            cooling_efficiency = np.zeros((NUM_PODS, num_steps))
            
            plasma_active = np.zeros(NUM_PODS, dtype=bool)
            plasma_power = np.zeros(NUM_PODS)
            activation_count = np.zeros(NUM_PODS)
            total_cooling_time = np.zeros(NUM_PODS)
            
            time_vec = np.arange(num_steps) * dt
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for step in range(num_steps):
                t = time_vec[step]
                
                for pod in range(NUM_PODS):
                    T = T_pods[:, :, :, pod].copy()
                    
                    workload_variation = workload_base[pod] + \
                                       workload_amplitude[pod] * np.sin(t * workload_frequency[pod] + workload_phases[pod])
                    
                    if pod in [2, 6]:
                        if np.random.random() < 0.05:
                            workload_variation += 0.4 * np.random.random()
                    
                    if pod in [5, 7]:
                        workload_variation = max(workload_variation, 1.2)
                    
                    workload_variation = np.clip(workload_variation, 0.3, 1.5)
                    
                    Q_gen = np.zeros(T.shape)
                    Q_gen[processor_zones] = 750 * workload_variation
                    
                    current_max_temp = np.max(T)
                    current_avg_temp = np.mean(T)
                    
                    prev_status = plasma_active[pod]
                    
                    if current_max_temp >= T_critical:
                        plasma_active[pod] = True
                        plasma_power[pod] = 1.0
                    elif plasma_active[pod]:
                        if current_avg_temp <= T_plasma_off:
                            plasma_active[pod] = False
                            plasma_power[pod] = max(0, plasma_power[pod] - 0.15)
                        else:
                            plasma_power[pod] = min(1.0, plasma_power[pod] + 0.12)
                    else:
                        if current_max_temp >= T_plasma_on:
                            plasma_active[pod] = True
                            plasma_power[pod] = 0.1
                        else:
                            plasma_power[pod] = max(0, plasma_power[pod] - 0.08)
                    
                    if plasma_active[pod] and not prev_status:
                        activation_count[pod] += 1
                    
                    if plasma_power[pod] > 0.5:
                        total_cooling_time[pod] += dt
                    
                    temp_excess = max(0, current_max_temp - T_plasma_off)
                    temp_range = max_temps_initial[pod] - T_plasma_off
                    cooling_factor = plasma_power[pod] * min(1, temp_excess / temp_range)
                    
                    u = u_base * cooling_factor
                    v = v_base * cooling_factor
                    w = w_base * cooling_factor
                    velocity_mag = np.sqrt(u**2 + v**2 + w**2)
                    
                    laplacian_T = laplace(T) / (dx**2)
                    
                    dTdx = np.gradient(T, dx, axis=0)
                    dTdy = np.gradient(T, dx, axis=1)
                    dTdz = np.gradient(T, dx, axis=2)
                    
                    advection = u * dTdx + v * dTdy + w * dTdz
                    
                    plasma_cooling_boost = 3.8
                    cooling_rate = velocity_mag * 1.3 * plasma_cooling_boost
                    natural_cooling = 0.06 * (T - T_ambient)
                    
                    dT = dt * (alpha * laplacian_T * 1.6 
                              - advection 
                              + Q_gen / (rho_air * cp_air)
                              - cooling_rate
                              - natural_cooling)
                    
                    T = T + dT
                    T = np.clip(T, T_ambient, 105)
                    
                    T_pods[:, :, :, pod] = T
                    
                    max_temps[pod, step] = current_max_temp
                    avg_temps[pod, step] = current_avg_temp
                    plasma_status[pod, step] = plasma_power[pod]
                    power_consumption[pod, step] = plasma_power[pod] * 5000
                    
                    if max_temps_initial[pod] > T_ambient:
                        cooling_efficiency[pod, step] = ((max_temps_initial[pod] - current_avg_temp) / 
                                                         (max_temps_initial[pod] - T_ambient) * 100)
                
                progress_bar.progress((step + 1) / num_steps)
                status_text.text(f"Simulating... Step {step + 1}/{num_steps} | Active pods: {np.sum(plasma_power > 0.5)}/{NUM_PODS}")
            
            progress_bar.empty()
            status_text.empty()
            
            return {
                'T_pods': T_pods,
                'max_temps': max_temps,
                'avg_temps': avg_temps,
                'plasma_status': plasma_status,
                'power_consumption': power_consumption,
                'cooling_efficiency': cooling_efficiency,
                'time_vec': time_vec,
                'activation_count': activation_count,
                'total_cooling_time': total_cooling_time,
                'plasma_power': plasma_power
            }
        
        # Run simulation button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Run Plasma Simulation", type="primary", use_container_width=True, key='run_plasma'):
                with st.spinner("Initializing plasma system..."):
                    T_pods, processor_zones, u_base, v_base, w_base, X, Y, Z = initialize_plasma_simulation()
                
                with st.spinner("Running plasma cooling simulation..."):
                    st.session_state.plasma_results = run_plasma_simulation(T_pods, processor_zones, u_base, v_base, w_base)
                    st.session_state.plasma_initialized = True
                
                st.success("Simulation completed successfully!")
        
        # Display results if simulation has been run
        if st.session_state.plasma_initialized and st.session_state.plasma_results is not None:
            results = st.session_state.plasma_results
            time_vec = results['time_vec']
            max_temps = results['max_temps']
            plasma_status = results['plasma_status']
            cooling_efficiency = results['cooling_efficiency']
            activation_count = results['activation_count']
            total_cooling_time = results['total_cooling_time']
            power_consumption = results['power_consumption']
            T_pods = results['T_pods']
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Pod Temperatures", "Plasma Control", "Performance Metrics", "Data Export"])
            
            with tab1:
                st.subheader("Pod Temperature Distributions (Mid-plane)")
                
                cols = st.columns(5)
                for pod in range(NUM_PODS):
                    col_idx = pod % 5
                    with cols[col_idx]:
                        mid_slice = T_pods[:, :, grid_res // 2, pod]
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=mid_slice,
                            colorscale='Jet',
                            zmin=T_ambient,
                            zmax=100,
                            colorbar=dict(title="°C")
                        ))
                        
                        status_color = 'red' if max_temps[pod, -1] >= T_critical else \
                                      'cyan' if results['plasma_power'][pod] > 0.5 else \
                                      'yellow' if results['plasma_power'][pod] > 0.1 else 'gray'
                        
                        fig.update_layout(
                            title=dict(
                                text=f"Pod {pod + 1}: {max_temps[pod, -1]:.1f}°C<br>{pod_scenarios[pod][:15]}",
                                font=dict(size=9, color=status_color)
                            ),
                            height=220,
                            margin=dict(l=5, r=5, t=50, b=5)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                st.subheader("Temperature Evolution - All Pods")
                fig = go.Figure()
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                for pod in range(NUM_PODS):
                    fig.add_trace(go.Scatter(
                        x=time_vec,
                        y=max_temps[pod, :],
                        mode='lines',
                        name=f'Pod {pod + 1}',
                        line=dict(color=colors[pod], width=2)
                    ))
                
                fig.add_hline(y=T_critical, line_dash="dash", line_color="red", 
                             annotation_text="Critical", annotation_position="right")
                fig.add_hline(y=T_plasma_on, line_dash="dash", line_color="orange",
                             annotation_text="Activate", annotation_position="right")
                fig.add_hline(y=T_plasma_off, line_dash="dash", line_color="blue",
                             annotation_text="Deactivate", annotation_position="right")
                
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Temperature (°C)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Plasma Power Timeline")
                
                fig = go.Figure()
                
                for pod in range(NUM_PODS):
                    fig.add_trace(go.Scatter(
                        x=time_vec,
                        y=plasma_status[pod, :] * 100,
                        mode='lines',
                        name=f'Pod {pod + 1}',
                        line=dict(color=colors[pod], width=2),
                        fill='tozeroy',
                        fillcolor=colors[pod],
                        opacity=0.3
                    ))
                
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Plasma Power (%)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Plasma Activation Heatmap")
                
                fig = go.Figure(data=go.Heatmap(
                    z=plasma_status * 100,
                    x=time_vec,
                    y=[f'Pod {i+1}' for i in range(NUM_PODS)],
                    colorscale='Hot',
                    colorbar=dict(title="Power %")
                ))
                
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Pod Number",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Energy Consumption")
                    total_energy = np.sum(power_consumption, axis=1) * dt / 3600
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Pod {i+1}' for i in range(NUM_PODS)],
                            y=total_energy,
                            marker_color='lightblue',
                            text=[f'{e:.1f} Wh' for e in total_energy],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        yaxis_title="Energy (Wh)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Plasma Activation Cycles")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Pod {i+1}' for i in range(NUM_PODS)],
                            y=activation_count,
                            marker_color='orange',
                            text=[f'{int(c)}' for c in activation_count],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        yaxis_title="Activation Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Average Cooling Efficiency")
                    avg_efficiency = np.mean(cooling_efficiency, axis=1)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Pod {i+1}' for i in range(NUM_PODS)],
                            y=avg_efficiency,
                            marker_color=colors,
                            text=[f'{e:.1f}%' for e in avg_efficiency],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        yaxis_title="Efficiency (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Plasma Active Time")
                    cooling_time_pct = (total_cooling_time / time_vec[-1]) * 100
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Pod {i+1}' for i in range(NUM_PODS)],
                            y=cooling_time_pct,
                            marker_color='lightgreen',
                            text=[f'{p:.1f}%' for p in cooling_time_pct],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        yaxis_title="Active Time (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("System Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Energy", f"{np.sum(total_energy):.1f} Wh")
                with col2:
                    st.metric("Avg Efficiency", f"{np.mean(avg_efficiency):.1f}%")
                with col3:
                    st.metric("Total Activations", f"{int(np.sum(activation_count))}")
                with col4:
                    st.metric("Simulation Time", f"{time_vec[-1]:.1f} s")
            
            with tab4:
                st.subheader("Export Simulation Data")
                
                summary_df = pd.DataFrame({
                    'Pod': [f'Pod {i+1}' for i in range(NUM_PODS)],
                    'Scenario': pod_scenarios,
                    'Initial Temp (°C)': max_temps[:, 0],
                    'Final Temp (°C)': max_temps[:, -1],
                    'Temp Reduction (°C)': max_temps[:, 0] - max_temps[:, -1],
                    'Activations': activation_count.astype(int),
                    'Active Time (%)': cooling_time_pct,
                    'Energy (Wh)': total_energy,
                    'Avg Efficiency (%)': avg_efficiency
                })
                
                st.dataframe(summary_df, use_container_width=True)
                
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv,
                    file_name="silahgrid_plasma_summary.csv",
                    mime="text/csv"
                )
        else:
            st.info("👆 Configure parameters in the sidebar and click 'Run Plasma Simulation' to start")
            
            st.subheader("System Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pod Scenarios:**")
                for i, scenario in enumerate(pod_scenarios):
                    st.write(f"Pod {i+1}: {scenario}")
            
            with col2:
                st.markdown("**Initial Temperatures:**")
                for i, temp in enumerate(max_temps_initial):
                    st.write(f"Pod {i+1}: {temp}°C")
    elif nav == 'Plasma Visualization':
        st.header('Plasma Visualization')
        st.info('Interactive EHD plasma demo (React) embedded below.')
        try:
            import streamlit.components.v1 as components
            from pathlib import Path
            page = Path.cwd() / 'plasma_pages' / 'plasma_viz.html'
            if page.exists():
                html = page.read_text(encoding='utf-8')
                components.html(html, height=900, scrolling=True)
            else:
                st.error('plasma_viz.html not found in plasma_pages/.')
        except Exception as e:
            st.error(f'Failed to embed Plasma Visualization: {e}')
            # Add an option to run the scaffold in-app without creating files
            if st.button('Run scaffold demo in app'):
                try:
                    import numpy as np
                    import matplotlib.pyplot as _plt
                except Exception:
                    st.error('NumPy and matplotlib are required to run the demo. Install with: pip install numpy matplotlib')
                else:
                    # Simple static demo: generate synthetic traces and render with matplotlib
                    num_pods = 8
                    num_steps = 70
                    time_vec = np.linspace(0, num_steps-1, num_steps)
                    np.random.seed(0)
                    base = np.array([52,88,68,78,48,90,65,92])
                    traces = np.zeros((num_pods, num_steps))
                    for i in range(num_pods):
                        traces[i] = base[i] + 5*np.sin(time_vec*0.1*(i+1)) + np.random.randn(num_steps)*1.5

                    fig = _plt.figure(figsize=(10,6))
                    for i in range(num_pods):
                        _plt.plot(time_vec, traces[i], label=f'Pod {i+1}')
                    _plt.xlabel('Step')
                    _plt.ylabel('Temp (C)')
                    _plt.title('Plasma Modulation - Temperature Traces (demo)')
                    _plt.legend()
                    _plt.grid(True)
                    _plt.tight_layout()
                    st.pyplot(fig)
                    _plt.close(fig)

# --- EFFICIENCY METRICS ---
# Only render efficiency metrics on the Home page (keep Map page clean)
if nav == 'Home':
    total_pod_energy = sum(pod.energy_used for pod in st.session_state.pods)
    total_dc_energy = st.session_state.data_center.energy_used
    total_energy_saved = sum(pod.energy_saved for pod in st.session_state.pods)
    st.metric("Total Pod Energy Used", f"{total_pod_energy:.1f}")
    st.metric("Total Data Center Energy Used", f"{total_dc_energy:.1f}")
    st.metric("Estimated Energy Saved", f"{total_energy_saved:.1f}")

# --- RUN SIMULATION ---    source venv/Scripts/activate
# Auto-run simulation every second
run_simulation()
time.sleep(1.0 / SPEED)
safe_rerun()

st.caption("Prototype by The She-nanigans — Innovation Hackathon 2025")
