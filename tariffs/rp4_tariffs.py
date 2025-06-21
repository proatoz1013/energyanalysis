# RP4 tariff structure and helper functions will be defined here.

# Example stub:
def get_tariff_data():
    """Return the RP4 tariff data structure."""
    return {
        "Non Domestic": [
            {
                "Tariff": "Low Voltage General",
                "Voltage": "Low Voltage",
                "Base Rate": 0.2703,          # RM/kWh
                "Capacity Rate": 0.0883,      # RM/kWh
                "Network Rate": 0.1482,       # RM/kWh
                "Retail Rate": 20.00,         # RM/month
                "Split": False,
                "Tiered": False
            },
            {
                "Tariff": "Low Voltage TOU",
                "Voltage": "Low Voltage",
                "Peak Rate": 0.2852,
                "OffPeak Rate": 0.2443,
                "Capacity Rate": 0.0883,      # RM/kWh
                "Network Rate": 0.1482,       # RM/kWh
                "Retail Rate": 20.00,
                "Split": True,
                "Tiered": False
            },
            {
                "Tariff": "Medium Voltage General",
                "Voltage": "Medium Voltage",
                "Base Rate": 0.2983,
                "Capacity Rate": 29.43,       # RM/kW
                "Network Rate": 59.84,        # RM/kW
                "Retail Rate": 200.00,
                "Split": False,
                "Tiered": False
            },
            {
                "Tariff": "Medium Voltage TOU",
                "Voltage": "Medium Voltage",
                "Peak Rate": 0.3132,
                "OffPeak Rate": 0.2723,
                "Capacity Rate": 30.19,       # RM/kW (peak)
                "Network Rate": 66.87,        # RM/kW (peak)
                "Retail Rate": 200.00,
                "Split": True,
                "Tiered": False
            },
            {
                "Tariff": "High Voltage General",
                "Voltage": "High Voltage",
                "Base Rate": 0.4303,
                "Capacity Rate": 16.68,
                "Network Rate": 14.53,
                "Retail Rate": 250.00,
                "Split": False,
                "Tiered": False
            },
            {
                "Tariff": "High Voltage TOU",
                "Voltage": "High Voltage",
                "Peak Rate": 0.4452,
                "OffPeak Rate": 0.4043,
                "Capacity Rate": 21.76,
                "Network Rate": 23.06,
                "Retail Rate": 250.00,
                "Split": True,
                "Tiered": False
            }
        ]
    }
