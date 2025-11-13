#!/usr/bin/env python3
"""
Test script for the new simplified event processing logic.

This script tests the TriggerEvents.process_historical_events() method
with the simplified approach:
- Trigger = target_series (monthly target)  
- Event = any timestamp where MD excess > 0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from smart_conservation import TriggerEvents

def test_simplified_event_processing():
    """Test the new simplified event processing logic."""
    
    print("🧪 Testing Simplified Event Processing Logic")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    # Create sample power consumption with some peaks
    base_power = 1000  # 1000 kW base load
    power_data = []
    
    for i, timestamp in enumerate(dates):
        # Add some variation and occasional peaks
        if i in [20, 21, 22, 45, 46, 60, 61, 62, 63]:  # Simulate events
            power = base_power + np.random.uniform(200, 500)  # Exceed target
        else:
            power = base_power + np.random.uniform(-100, 100)  # Normal variation
        power_data.append(power)
    
    # Create DataFrame
    df_sim = pd.DataFrame({
        'timestamp': dates,
        'kw_15min': power_data
    })
    df_sim.set_index('timestamp', inplace=True)
    
    # Create target series (monthly target)
    target_kw = 1200  # 1200 kW target
    target_series = pd.Series([target_kw] * len(dates), index=dates)
    
    # Create config_data
    config_data = {
        'df_sim': df_sim,
        'power_col': 'kw_15min',
        'target_series': target_series
    }
    
    print(f"📊 Sample Data Created:")
    print(f"   Timestamps: {len(dates)}")
    print(f"   Power column: kw_15min")
    print(f"   Target: {target_kw} kW")
    print(f"   Expected events at indices: [20-22, 45-46, 60-63]")
    print()
    
    # Test the new method
    try:
        print("🔧 Testing TriggerEvents.process_historical_events()...")
        
        trigger_events = TriggerEvents()
        enhanced_df = trigger_events.process_historical_events(config_data)
        
        print("✅ Method executed successfully!")
        print()
        
        # Analyze results
        total_timestamps = len(enhanced_df)
        event_timestamps = enhanced_df['is_event'].sum()
        unique_events = enhanced_df[enhanced_df['event_id'] > 0]['event_id'].nunique()
        max_excess = enhanced_df['excess_demand_kw'].max()
        min_excess = enhanced_df['excess_demand_kw'].min()
        
        print(f"📈 Analysis Results:")
        print(f"   Total timestamps: {total_timestamps}")
        print(f"   Event timestamps: {event_timestamps}")
        print(f"   Unique events detected: {unique_events}")
        print(f"   Max excess: {max_excess:.2f} kW")
        print(f"   Min excess: {min_excess:.2f} kW")
        print()
        
        # Show event details
        event_data = enhanced_df[enhanced_df['is_event']].copy()
        if not event_data.empty:
            print(f"🎯 Event Details:")
            print(f"   Event indices: {event_data.index.tolist()[:10]}...")  # Show first 10
            print(f"   Event IDs: {event_data['event_id'].unique()}")
            print(f"   Event starts: {event_data[event_data['event_start']].index.tolist()}")
            print()
            
            # Show first few event rows
            print("📋 Sample Event Data:")
            display_cols = ['excess_demand_kw', 'is_event', 'event_id', 'event_start', 'event_duration']
            print(event_data[display_cols].head(10).to_string())
            print()
        
        # Show sample of all data
        print("📋 Sample Enhanced DataFrame:")
        display_cols = ['kw_15min', 'excess_demand_kw', 'is_event', 'event_id', 'event_start']
        print(enhanced_df[display_cols].head(30).to_string())
        
        print("\n" + "=" * 50)
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simplified_event_processing()