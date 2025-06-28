#!/usr/bin/env python3

import sys
import traceback

print("Testing md_shaving_battery import...")

try:
    print("1. Importing module...")
    import md_shaving_battery
    print("   ✓ Module imported successfully")
    
    print("2. Checking module attributes...")
    attrs = dir(md_shaving_battery)
    print(f"   Available attributes: {attrs}")
    
    print("3. Looking for md_shaving_battery_page function...")
    if hasattr(md_shaving_battery, 'md_shaving_battery_page'):
        print("   ✓ Function found!")
        func = getattr(md_shaving_battery, 'md_shaving_battery_page')
        print(f"   Function type: {type(func)}")
    else:
        print("   ✗ Function NOT found")
        
    print("4. Testing direct import...")
    try:
        from md_shaving_battery import md_shaving_battery_page
        print("   ✓ Direct import successful")
    except ImportError as e:
        print(f"   ✗ Direct import failed: {e}")
        
except Exception as e:
    print(f"Error during import: {e}")
    traceback.print_exc()
