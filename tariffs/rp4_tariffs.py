# RP4 tariff structure and helper functions will be defined here.

# Example stub:
def get_tariff_data():
    """Return the RP4 tariff data structure."""
    return {
        "Non Domestic": {
            "Tariffs": [
                {
                    "Tariff": "Low Voltage General",
                    "Voltage": "Low Voltage",
                    "Type": "General",
                    "Energy Rate": 0.2703,
                    "Capacity Rate": 0.0883,
                    "Network Rate": 0.1482,
                    "Retail Rate": 20.00,
                    "Unit": {
                        "Energy Rate": "RM/kWh",
                        "Capacity Rate": "RM/kWh",
                        "Network Rate": "RM/kWh",
                        "Retail Rate": "RM/month"
                    },
                    "Split": False,
                    "Tiered": False
                },
                {
                    "Tariff": "Low Voltage TOU",
                    "Voltage": "Low Voltage",
                    "Type": "TOU",
                    "Peak Rate": 0.2852,
                    "OffPeak Rate": 0.2443,
                    "Capacity Rate": 0.0883,
                    "Network Rate": 0.1482,
                    "Retail Rate": 20.00,
                    "Unit": {
                        "Peak Rate": "RM/kWh",
                        "OffPeak Rate": "RM/kWh",
                        "Capacity Rate": "RM/kWh",
                        "Network Rate": "RM/kWh",
                        "Retail Rate": "RM/month"
                    },
                    "Split": True,
                    "Tiered": False
                },
                {
                    "Tariff": "Medium Voltage General",
                    "Voltage": "Medium Voltage",
                    "Type": "General",
                    "Energy Rate": 0.2983,
                    "Capacity Rate": 29.43,
                    "Network Rate": 59.84,
                    "Retail Rate": 200.00,
                    "Unit": {
                        "Energy Rate": "RM/kWh",
                        "Capacity Rate": "RM/kW",
                        "Network Rate": "RM/kW",
                        "Retail Rate": "RM/month"
                    },
                    "Split": False,
                    "Tiered": False
                },
                {
                    "Tariff": "Medium Voltage TOU",
                    "Voltage": "Medium Voltage",
                    "Type": "TOU",
                    "Peak Rate": 0.3132,
                    "OffPeak Rate": 0.2723,
                    "Capacity Rate": 30.19,
                    "Network Rate": 66.87,
                    "Retail Rate": 200.00,
                    "Unit": {
                        "Peak Rate": "RM/kWh",
                        "OffPeak Rate": "RM/kWh",
                        "Capacity Rate": "RM/kW",
                        "Network Rate": "RM/kW",
                        "Retail Rate": "RM/month"
                    },
                    "Split": True,
                    "Tiered": False
                },
                {
                    "Tariff": "High Voltage General",
                    "Voltage": "High Voltage",
                    "Type": "General",
                    "Energy Rate": 0.4303,
                    "Capacity Rate": 16.68,
                    "Network Rate": 14.53,
                    "Retail Rate": 250.00,
                    "Unit": {
                        "Energy Rate": "RM/kWh",
                        "Capacity Rate": "RM/kW",
                        "Network Rate": "RM/kW",
                        "Retail Rate": "RM/month"
                    },
                    "Split": False,
                    "Tiered": False
                },
                {
                    "Tariff": "High Voltage TOU",
                    "Voltage": "High Voltage",
                    "Type": "TOU",
                    "Peak Rate": 0.4452,
                    "OffPeak Rate": 0.4043,
                    "Capacity Rate": 21.76,
                    "Network Rate": 23.06,
                    "Retail Rate": 250.00,
                    "Unit": {
                        "Peak Rate": "RM/kWh",
                        "OffPeak Rate": "RM/kWh",
                        "Capacity Rate": "RM/kW",
                        "Network Rate": "RM/kW",
                        "Retail Rate": "RM/month"
                    },
                    "Split": True,
                    "Tiered": False
                }
            ]
        }
    }
