import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.ndimage import laplace
import time
import sys
from pathlib import Path

# Add parent directory to path to import pod_config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pod_config import (
    NUM_PODS, POD_SCENARIOS, PLASMA_INITIAL_TEMPS, 
    POD_DIMENSIONS, WORKLOAD_CHARACTERISTICS
)

# Page configuration
st.set_page_config(
    page_title="SilahGrid Plasma Cooling System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Plasma Cooling Network Metrics</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Simulation Controls")

# System parameters
st.sidebar.subheader("System Configuration")
num_pods = NUM_PODS
num_steps = st.sidebar.slider("Simulation Steps", 20, 100, 70, 10)
dt = st.sidebar.slider("Time Step (s)", 1.0, 10.0, 4.2, 0.5)
grid_res = st.sidebar.slider("Grid Resolution", 10, 20, 15, 1)

# Temperature thresholds
st.sidebar.subheader("Temperature Thresholds (°C)")
T_ambient = st.sidebar.number_input("Ambient Temperature", 20, 50, 45)
T_plasma_on = st.sidebar.number_input("Plasma Activation", 40, 70, 55)
T_plasma_off = st.sidebar.number_input("Plasma Deactivation", 35, 60, 45)
T_critical = st.sidebar.number_input("Critical Temperature", 60, 90, 75)

# Pod scenarios from centralized config
pod_scenarios = POD_SCENARIOS

# Initial temperatures from centralized config
max_temps_initial = PLASMA_INITIAL_TEMPS

# Workload characteristics from centralized config
workload_base = WORKLOAD_CHARACTERISTICS['base']
workload_amplitude = WORKLOAD_CHARACTERISTICS['amplitude']
workload_frequency = WORKLOAD_CHARACTERISTICS['frequency']
workload_phases = WORKLOAD_CHARACTERISTICS['phases']

# Physical constants
k_air = 0.028
rho_air = 1.2
cp_air = 1005
alpha = k_air / (rho_air * cp_air)

# Pod dimensions from centralized config
pod_length = POD_DIMENSIONS['length']
pod_width = POD_DIMENSIONS['width']
pod_height = POD_DIMENSIONS['height']

def initialize_simulation():
    """Initialize the simulation state"""
    # Create 3D grid
    X, Y, Z = np.meshgrid(
        np.linspace(0, pod_length, grid_res),
        np.linspace(0, pod_width, grid_res),
        np.linspace(0, pod_height, grid_res)
    )
    
    # Processor zones
    processor_zones = np.zeros(X.shape, dtype=bool)
    processor_zones[5:10, 7:12, 9:14] = True
    processor_zones[11:14, 7:12, 7:11] = True
    
    # Initialize temperature for all pods
    T_pods = np.ones((grid_res, grid_res, grid_res, num_pods)) * T_ambient
    
    for i in range(num_pods):
        T = T_pods[:, :, :, i].copy()
        T[processor_zones] = max_temps_initial[i]
        
        # Add spatial temperature gradient
        for z in range(grid_res):
            heat_factor = (z / grid_res) ** 2
            T[:, :, z] += heat_factor * (max_temps_initial[i] - T_ambient) * 0.3
        
        T_pods[:, :, :, i] = T
    
    # EHD airflow patterns
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

def run_simulation(T_pods, processor_zones, u_base, v_base, w_base):
    """Run the plasma cooling simulation"""
    
    dx = pod_length / (grid_res - 1)
    
    # Storage arrays
    max_temps = np.zeros((num_pods, num_steps))
    avg_temps = np.zeros((num_pods, num_steps))
    plasma_status = np.zeros((num_pods, num_steps))
    power_consumption = np.zeros((num_pods, num_steps))
    cooling_efficiency = np.zeros((num_pods, num_steps))
    
    plasma_active = np.zeros(num_pods, dtype=bool)
    plasma_power = np.zeros(num_pods)
    activation_count = np.zeros(num_pods)
    total_cooling_time = np.zeros(num_pods)
    
    time_vec = np.arange(num_steps) * dt
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(num_steps):
        t = time_vec[step]
        
        for pod in range(num_pods):
            T = T_pods[:, :, :, pod].copy()
            
            # Workload modeling
            workload_variation = workload_base[pod] + \
                               workload_amplitude[pod] * np.sin(t * workload_frequency[pod] + workload_phases[pod])
            
            # Add random spikes for burst scenarios
            if pod in [2, 6]:
                if np.random.random() < 0.05:
                    workload_variation += 0.4 * np.random.random()
            
            # Critical pods stay hot
            if pod in [5, 7]:
                workload_variation = max(workload_variation, 1.2)
            
            workload_variation = np.clip(workload_variation, 0.3, 1.5)
            
            Q_gen = np.zeros(T.shape)
            Q_gen[processor_zones] = 750 * workload_variation
            
            # Temperature metrics
            current_max_temp = np.max(T)
            current_avg_temp = np.mean(T)
            
            # Intelligent plasma control
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
            
            # Count activations
            if plasma_active[pod] and not prev_status:
                activation_count[pod] += 1
            
            if plasma_power[pod] > 0.5:
                total_cooling_time[pod] += dt
            
            # Adaptive cooling
            temp_excess = max(0, current_max_temp - T_plasma_off)
            temp_range = max_temps_initial[pod] - T_plasma_off
            cooling_factor = plasma_power[pod] * min(1, temp_excess / temp_range)
            
            u = u_base * cooling_factor
            v = v_base * cooling_factor
            w = w_base * cooling_factor
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            
            # Heat transfer physics
            laplacian_T = laplace(T) / (dx**2)
            
            # Gradient calculation (simplified)
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
            
            # Record metrics
            max_temps[pod, step] = current_max_temp
            avg_temps[pod, step] = current_avg_temp
            plasma_status[pod, step] = plasma_power[pod]
            power_consumption[pod, step] = plasma_power[pod] * 5000
            
            if max_temps_initial[pod] > T_ambient:
                cooling_efficiency[pod, step] = ((max_temps_initial[pod] - current_avg_temp) / 
                                                 (max_temps_initial[pod] - T_ambient) * 100)
        
        # Update progress
        progress_bar.progress((step + 1) / num_steps)
        status_text.text(f"Simulating... Step {step + 1}/{num_steps} | Active pods: {np.sum(plasma_power > 0.5)}/{num_pods}")
    
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

def create_visualizations(results):
    """Create comprehensive visualizations"""
    
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
        # Pod temperature heatmaps
        st.subheader("Pod Temperature Distributions (Mid-plane)")
        
        cols = st.columns(4)
        for pod in range(num_pods):
            col_idx = pod % 4
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
                        text=f"Pod {pod + 1}: {max_temps[pod, -1]:.1f}°C<br>{pod_scenarios[pod]}",
                        font=dict(size=10, color=status_color)
                    ),
                    height=250,
                    margin=dict(l=10, r=10, t=60, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Temperature evolution
        st.subheader("Temperature Evolution - All Pods")
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for pod in range(num_pods):
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
        # Plasma activation timeline
        st.subheader("Plasma Power Timeline")
        
        fig = go.Figure()
        
        for pod in range(num_pods):
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
        
        # Plasma activation heatmap
        st.subheader("Plasma Activation Heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=plasma_status * 100,
            x=time_vec,
            y=[f'Pod {i+1}' for i in range(num_pods)],
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
            # Energy consumption
            st.subheader("Energy Consumption")
            total_energy = np.sum(power_consumption, axis=1) * dt / 3600
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'Pod {i+1}' for i in range(num_pods)],
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
            
            # Activation cycles
            st.subheader("Plasma Activation Cycles")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'Pod {i+1}' for i in range(num_pods)],
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
            # Cooling efficiency
            st.subheader("Average Cooling Efficiency")
            avg_efficiency = np.mean(cooling_efficiency, axis=1)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'Pod {i+1}' for i in range(num_pods)],
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
            
            # Active time percentage
            st.subheader("Plasma Active Time")
            cooling_time_pct = (total_cooling_time / time_vec[-1]) * 100
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f'Pod {i+1}' for i in range(num_pods)],
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
        
        # Summary metrics
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
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'Pod': [f'Pod {i+1}' for i in range(num_pods)],
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
        
        # Download button
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary CSV",
            data=csv,
            file_name="silahgrid_summary.csv",
            mime="text/csv"
        )

# Main app logic
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
    st.session_state.results = None

# Run simulation button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Initializing system..."):
            T_pods, processor_zones, u_base, v_base, w_base, X, Y, Z = initialize_simulation()
        
        with st.spinner("Running plasma cooling simulation..."):
            st.session_state.results = run_simulation(T_pods, processor_zones, u_base, v_base, w_base)
            st.session_state.simulation_run = True
        
        st.success("Simulation completed successfully!")

# Display results if simulation has been run
if st.session_state.simulation_run and st.session_state.results is not None:
    create_visualizations(st.session_state.results)
else:
    st.info("👆 Configure parameters in the sidebar and click 'Run Simulation' to start")
    
    # Show system overview
    st.subheader("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Pod Scenarios:*")
        for i, scenario in enumerate(pod_scenarios):
            st.write(f"Pod {i+1}: {scenario}")
    
    with col2:
        st.markdown("*Initial Temperatures:*")
        for i, temp in enumerate(max_temps_initial):
            st.write(f"Pod {i+1}: {temp}°C")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>SilahGrid Intelligent Plasma Cooling System | Zero Water Consumption | AI-Optimized Thermal Management</p>
</div>
""", unsafe_allow_html=True)