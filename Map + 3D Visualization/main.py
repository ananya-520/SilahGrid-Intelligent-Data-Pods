# # main.py
# from simulation_env import run_simulation
# from visualization import create_pod_map

# if __name__ == "__main__":
#     print("=== Smart Edge AI Pods Simulation – Task 1 ===")

#     # Step 1: run short simulation to generate pod states
#     pods = run_simulation(num_pods=10, duration=3)

#     # Step 2: visualize pod placement on a map
#     create_pod_map(pods)

#     print("=== Task 1 complete. Open smart_pods_map.html to view pods. ===")



# main.py
from simulation_env import run_simulation
from visualization import create_pod_map

if __name__ == "__main__":
    print("=== Smart Edge AI Pods Simulation – Task 1 ===")

    # Step 1: run short simulation to generate pod states
    pods = run_simulation(num_pods=10, duration=3)

    # Step 2: visualize pod placement on a map with 3D view
    create_pod_map(pods)

    print("=== Task 1 complete. Open smart_pods_map.html to view pods. ===")
