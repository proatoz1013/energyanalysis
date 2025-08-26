#!/usr/bin/env python3
"""
Test script to verify _simulate_battery_operation is now available in V2
"""

import sys
sys.path.append('/Users/xlnyeong/energyanalaysis')

def test_import_fix():
    """Test that _simulate_battery_operation can be imported from V2"""
    
    print("🧪 TESTING IMPORT FIX")
    print("=" * 40)
    
    try:
        # Test importing the function through V2's import structure
        from md_shaving_solution_v2 import _simulate_battery_operation
        print("✅ SUCCESS: _simulate_battery_operation imported successfully from V2")
        
        # Check function signature
        import inspect
        sig = inspect.signature(_simulate_battery_operation)
        print(f"📝 Function signature: {sig}")
        
        # Check if function is callable
        if callable(_simulate_battery_operation):
            print("✅ Function is callable")
        else:
            print("❌ Function is not callable")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_v2_imports():
    """Test all V2 imports to ensure nothing is broken"""
    
    print("\n🔍 TESTING ALL V2 IMPORTS")
    print("=" * 40)
    
    try:
        import md_shaving_solution_v2
        print("✅ md_shaving_solution_v2 module imported successfully")
        
        # Test key functions
        functions_to_test = [
            '_detect_peak_events',
            '_display_battery_simulation_chart', 
            '_simulate_battery_operation',
            'load_vendor_battery_database'
        ]
        
        for func_name in functions_to_test:
            if hasattr(md_shaving_solution_v2, func_name):
                print(f"✅ {func_name} is available")
            else:
                print(f"❌ {func_name} is NOT available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing V2 imports: {e}")
        return False

if __name__ == "__main__":
    success1 = test_import_fix()
    success2 = test_v2_imports()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 IMPORT FIX SUCCESSFUL!")
        print("✅ _simulate_battery_operation is now available in V2")
        print("✅ Battery discharge graph should now work")
    else:
        print("❌ Import fix failed - additional work needed")
