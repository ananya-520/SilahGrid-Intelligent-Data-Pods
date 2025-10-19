"""
Generate consistent pod HTML pages based on centralized pod_config
"""
from pod_config import DUBAI_LOCATIONS, POD_INITIAL_VALUES, POD_SCENARIOS
from pathlib import Path

# Skybox URLs for each pod (reusing as needed)
SKYBOX_URLS = [
    "https://skybox.blockadelabs.com/e/a65de74e75260778cb509f6b4fe227f9",
    "https://skybox.blockadelabs.com/e/b72ef85e86371889dc610f7c5fg338a0",
    "https://skybox.blockadelabs.com/e/c83fg96f97482990ed721g8d6gh449b1",
    "https://skybox.blockadelabs.com/e/d94gh07g08593aa1fe832h9e7hi550c2",
    "https://skybox.blockadelabs.com/e/e05hi18h19604bb2gf943i0f8ij661d3",
    "https://skybox.blockadelabs.com/e/a65de74e75260778cb509f6b4fe227f9",
    "https://skybox.blockadelabs.com/e/b72ef85e86371889dc610f7c5fg338a0",
    "https://skybox.blockadelabs.com/e/c83fg96f97482990ed721g8d6gh449b1",
    "https://skybox.blockadelabs.com/e/d94gh07g08593aa1fe832h9e7hi550c2",
    "https://skybox.blockadelabs.com/e/e05hi18h19604bb2gf943i0f8ij661d3"
]

def generate_pod_html(pod_id):
    """Generate HTML for a specific pod with consistent initial values"""
    
    # Get pod data (pod_id is 1-indexed)
    idx = pod_id - 1
    location_name = DUBAI_LOCATIONS[idx][0]
    cpu_load = POD_INITIAL_VALUES['cpu_load'][idx]
    temperature = POD_INITIAL_VALUES['temperature'][idx]
    battery = POD_INITIAL_VALUES['battery'][idx]
    solar_power = POD_INITIAL_VALUES['solar_power'][idx]
    ozone_level = POD_INITIAL_VALUES['ozone_level'][idx]
    scenario = POD_SCENARIOS[idx]
    skybox_url = SKYBOX_URLS[idx]
    
    # Determine status based on values
    if battery < 30:
        status = "Low Power Mode"
    elif temperature > 45:
        status = "High Temperature - Cooling"
    elif cpu_load > 70:
        status = f"{scenario}"
    else:
        status = "Idle"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{location_name} 3D Visualization</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{
    margin: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    display: flex;
    height: 100vh;
    background: radial-gradient(circle at 30% 50%, #0a0a0a 0%, #000 80%);
    color: #fff;
  }}
  #threejs-container {{
    width: 65%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #000;
    border-right: 2px solid #1e1e1e;
  }}
  iframe {{
    border: none;
    border-radius: 10px;
    width: 95%;
    height: 95%;
    box-shadow: 0 0 30px rgba(0,0,0,0.9);
  }}
  #pod-info {{
    width: 35%;
    padding: 30px;
    background-color: #141414;
    overflow-y: auto;
  }}
  h2 {{
    font-size: 1.8em;
    color: #4cc9f0;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }}
  .status {{
    font-size: 1.2em;
    font-weight: bold;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
    text-align: center;
    background: linear-gradient(90deg, #1a1a1a, #2a2a2a);
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
  }}
  ul {{
    list-style: none;
    padding: 0;
  }}
  li {{
    background: #1e1e1e;
    margin: 10px 0;
    padding: 15px;
    border-radius: 10px;
    font-size: 1.05em;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.6);
  }}
  li span {{
    color: #4cc9f0;
    font-weight: bold;
  }}
  .chart-container {{
    background: #1a1a1a;
    padding: 15px;
    margin-top: 25px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
  }}
  canvas {{
    width: 100% !important;
    height: 220px !important;
  }}
</style>

<script>
let cpuChart, tempChart, batteryChart;
const MAX_POINTS = 20;

async function refreshData() {{
  try {{
    const response = await fetch("metrics.json?_=" + new Date().getTime());
    if (!response.ok) return;
    const data = await response.json();

    // Update status text
    document.getElementById("status").innerText = data["Status"];
    document.getElementById("cpu").innerText = data["CPU Load (%)"] + "%";
    document.getElementById("temp").innerText = data["Temperature (°C)"] + " °C";
    document.getElementById("battery").innerText = data["Battery (%)"] + "%";
    document.getElementById("solar").innerText = data["Solar Power (W)"] + " W";
    document.getElementById("ozone").innerText = data["Ozone (ppm)"] + " ppm";

    const timestamp = new Date().toLocaleTimeString();

    // Update charts
    addChartData(cpuChart, timestamp, data["CPU Load (%)"]);
    addChartData(tempChart, timestamp, data["Temperature (°C)"]);
    addChartData(batteryChart, timestamp, data["Battery (%)"]);
  }} catch (e) {{
    console.error(e);
  }}
}}

function addChartData(chart, label, value) {{
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > MAX_POINTS) {{
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }}
  chart.update();
}}

function initCharts() {{
  const ctxCpu = document.getElementById('cpuChart').getContext('2d');
  const ctxTemp = document.getElementById('tempChart').getContext('2d');
  const ctxBattery = document.getElementById('batteryChart').getContext('2d');

  cpuChart = new Chart(ctxCpu, {{
    type: 'line',
    data: {{
      labels: [],
      datasets: [{{
        label: 'CPU Load (%)',
        borderColor: '#ffb347',
        data: [],
        fill: false,
        tension: 0.3
      }}]
    }},
    options: {{ scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
  }});

  tempChart = new Chart(ctxTemp, {{
    type: 'line',
    data: {{
      labels: [],
      datasets: [{{
        label: 'Temperature (°C)',
        borderColor: '#ff5c8a',
        data: [],
        fill: false,
        tension: 0.3
      }}]
    }},
    options: {{ scales: {{ y: {{ beginAtZero: true, max: 60 }} }} }}
  }});

  batteryChart = new Chart(ctxBattery, {{
    type: 'line',
    data: {{
      labels: [],
      datasets: [{{
        label: 'Battery (%)',
        borderColor: '#00ff99',
        data: [],
        fill: false,
        tension: 0.3
      }}]
    }},
    options: {{ scales: {{ y: {{ beginAtZero: true, max: 100 }} }} }}
  }});
}}

window.onload = () => {{
  initCharts();
  refreshData();
  setInterval(refreshData, 3000);
}};
</script>

</head>
<body>
  <div id="threejs-container">
    <iframe src="{skybox_url}" allow="fullscreen"></iframe>
  </div>

  <div id="pod-info">
    <h2>{location_name} (ID: {pod_id})</h2>
    <div id="status" class="status">{status}</div>
    <ul>
      <li><strong>CPU Load:</strong> <span id="cpu">{cpu_load:.1f}%</span></li>
      <li><strong>Temperature:</strong> <span id="temp">{temperature:.1f} °C</span></li>
      <li><strong>Battery:</strong> <span id="battery">{battery:.1f}%</span></li>
      <li><strong>Solar Power:</strong> <span id="solar">{solar_power:.1f} W</span></li>
      <li><strong>Ozone Level:</strong> <span id="ozone">{ozone_level:.3f} ppm</span></li>
    </ul>

    <div class="chart-container">
      <canvas id="cpuChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="tempChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="batteryChart"></canvas>
    </div>
  </div>
</body>
</html>
"""
    return html_content

def generate_all_pod_pages():
    """Generate HTML pages for all pods"""
    pod_pages_dir = Path("pod_pages")
    pod_pages_dir.mkdir(exist_ok=True)
    
    for pod_id in range(1, 11):
        html_content = generate_pod_html(pod_id)
        output_file = pod_pages_dir / f"pod_{pod_id}.html"
        output_file.write_text(html_content, encoding='utf-8')
        print(f"Generated: {output_file}")
    
    print(f"\n✓ Successfully generated all 10 pod pages with consistent values!")

if __name__ == "__main__":
    generate_all_pod_pages()
