"""
Verification script to check if all pod values are consistent across the system
"""
from pod_config import (
    NUM_PODS, POD_INITIAL_VALUES, DUBAI_LOCATIONS, 
    POD_SCENARIOS, get_initial_value, get_pod_location
)
from pathlib import Path
import re

def verify_pod_pages():
    """Verify that all pod HTML pages have correct values"""
    print("=" * 60)
    print("VERIFYING POD HTML PAGES")
    print("=" * 60)
    
    pod_pages_dir = Path("pod_pages")
    all_correct = True
    
    for pod_id in range(1, NUM_PODS + 1):
        html_file = pod_pages_dir / f"pod_{pod_id}.html"
        
        if not html_file.exists():
            print(f"❌ Pod {pod_id}: File missing!")
            all_correct = False
            continue
        
        content = html_file.read_text(encoding='utf-8')
        
        # Expected values
        expected_cpu = POD_INITIAL_VALUES['cpu_load'][pod_id - 1]
        expected_temp = POD_INITIAL_VALUES['temperature'][pod_id - 1]
        expected_battery = POD_INITIAL_VALUES['battery'][pod_id - 1]
        expected_solar = POD_INITIAL_VALUES['solar_power'][pod_id - 1]
        expected_ozone = POD_INITIAL_VALUES['ozone_level'][pod_id - 1]
        expected_location = DUBAI_LOCATIONS[pod_id - 1][0]
        
        # Check values in HTML
        checks = [
            (f">{expected_cpu:.1f}%<", "CPU Load"),
            (f">{expected_temp:.1f} °C<", "Temperature"),
            (f">{expected_battery:.1f}%<", "Battery"),
            (f">{expected_solar:.1f} W<", "Solar Power"),
            (f">{expected_ozone:.3f} ppm<", "Ozone"),
            (f"<title>{expected_location}", "Location in title"),
            (f"<h2>{expected_location} (ID: {pod_id})", "Location in header")
        ]
        
        pod_correct = True
        for check_str, label in checks:
            if check_str not in content:
                print(f"❌ Pod {pod_id}: {label} mismatch!")
                pod_correct = False
                all_correct = False
        
        if pod_correct:
            print(f"✓ Pod {pod_id} ({expected_location}): All values correct!")
    
    print()
    return all_correct

def verify_config_consistency():
    """Verify that pod_config.py has consistent data"""
    print("=" * 60)
    print("VERIFYING POD_CONFIG.PY CONSISTENCY")
    print("=" * 60)
    
    all_correct = True
    
    # Check that all lists have the correct length
    for key, values in POD_INITIAL_VALUES.items():
        if len(values) != NUM_PODS:
            print(f"❌ {key}: Length mismatch! Expected {NUM_PODS}, got {len(values)}")
            all_correct = False
    
    if len(DUBAI_LOCATIONS) != NUM_PODS:
        print(f"❌ DUBAI_LOCATIONS: Length mismatch! Expected {NUM_PODS}, got {len(DUBAI_LOCATIONS)}")
        all_correct = False
    
    if len(POD_SCENARIOS) != NUM_PODS:
        print(f"❌ POD_SCENARIOS: Length mismatch! Expected {NUM_PODS}, got {len(POD_SCENARIOS)}")
        all_correct = False
    
    if all_correct:
        print("✓ All configuration lists have correct length!")
    
    # Test helper functions
    print("\nTesting helper functions:")
    for pod_id in range(1, NUM_PODS + 1):
        location = get_pod_location(pod_id)
        cpu = get_initial_value('cpu_load', pod_id)
        
        if location is None or cpu is None:
            print(f"❌ Pod {pod_id}: Helper functions returned None!")
            all_correct = False
    
    if all_correct:
        print("✓ All helper functions work correctly!")
    
    print()
    return all_correct

def show_pod_summary():
    """Display a summary table of all pod values"""
    print("=" * 100)
    print("POD VALUES SUMMARY")
    print("=" * 100)
    print(f"{'Pod':<4} {'Location':<30} {'CPU%':<7} {'Temp°C':<8} {'Solar W':<9} {'Batt%':<7} {'Scenario':<25}")
    print("-" * 100)
    
    for pod_id in range(1, NUM_PODS + 1):
        idx = pod_id - 1
        location = DUBAI_LOCATIONS[idx][0]
        cpu = POD_INITIAL_VALUES['cpu_load'][idx]
        temp = POD_INITIAL_VALUES['temperature'][idx]
        solar = POD_INITIAL_VALUES['solar_power'][idx]
        battery = POD_INITIAL_VALUES['battery'][idx]
        scenario = POD_SCENARIOS[idx]
        
        print(f"{pod_id:<4} {location:<30} {cpu:<7.1f} {temp:<8.1f} {solar:<9.1f} {battery:<7.1f} {scenario:<25}")
    
    print("=" * 100)
    print()

def main():
    """Run all verification checks"""
    print("\n" + "=" * 60)
    print("SILAHGRID POD VALUES VERIFICATION")
    print("=" * 60)
    print()
    
    # Show summary
    show_pod_summary()
    
    # Run checks
    config_ok = verify_config_consistency()
    pages_ok = verify_pod_pages()
    
    # Final result
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if config_ok and pages_ok:
        print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("All pod values are consistent across the system!")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please review the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
