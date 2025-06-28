#!/usr/bin/env python3

# Test file to check if md_shaving_battery_page can be imported
try:
    from md_shaving_battery import md_shaving_battery_page
    print("✅ Successfully imported md_shaving_battery_page")
    print(f"Function type: {type(md_shaving_battery_page)}")
    print(f"Function name: {md_shaving_battery_page.__name__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

# Let's also try to import the module itself
try:
    import md_shaving_battery
    print("✅ Successfully imported md_shaving_battery module")
    print(f"Module attributes: {dir(md_shaving_battery)}")
except Exception as e:
    print(f"❌ Module import error: {e}")
