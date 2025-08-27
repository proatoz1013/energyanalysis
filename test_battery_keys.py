#!/usr/bin/env python3
import json

# Load the database to verify the keys
with open('vendor_battery_database.json', 'r') as f:
    battery_db = json.load(f)

# Test one battery entry
test_battery = list(battery_db.values())[0]
print('Database keys (first 8):')
for key in list(test_battery.keys())[:8]:
    print(f'  {key}')

print()
print('Testing specific keys:')
print(f'  power_kW: {"✓" if "power_kW" in test_battery else "✗"} - Value: {test_battery.get("power_kW", "NOT FOUND")}')
print(f'  energy_kWh: {"✓" if "energy_kWh" in test_battery else "✗"} - Value: {test_battery.get("energy_kWh", "NOT FOUND")}')
print(f'  power_kw: {"✓" if "power_kw" in test_battery else "✗"} - Value: {test_battery.get("power_kw", "NOT FOUND")}')
print(f'  energy_kwh: {"✓" if "energy_kwh" in test_battery else "✗"} - Value: {test_battery.get("energy_kwh", "NOT FOUND")}')
