TARIFF_DATA = {
  "Residential": {
    "Tariff Groups": {
      "Domestic": {
        "Tariffs": [
          {
            "Tariff": "Domestic Block Tariff",
            "Rates": {},
            "Rules": {
              "tiered": True,
              "charge_capacity_by": None,
              "charge_network_by": None,
              "has_peak_split": False,
              "afa_applicable": False
            }
          }
        ]
      }
    }
  },
  "Business": {
    "Tariff Groups": {
      "Non Domestic": {
        "Tariffs": [
          {
            "Tariff": "Low Voltage General",
            "Voltage": "Low Voltage",
            "Type": "General",
            "Rates": {
              "Energy Rate": 0.2703,
              "Capacity Rate": 0.0883,
              "Network Rate": 0.1482,
              "Retail Rate": 20.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kWh",
              "charge_network_by": "kWh",
              "has_peak_split": False,
              "afa_applicable": False,
              "ktwbb_applicable": True
            }
          },
          {
            "Tariff": "Low Voltage TOU",
            "Voltage": "Low Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.2852,
              "OffPeak Rate": 0.2443,
              "Capacity Rate": 0.0883,
              "Network Rate": 0.1482,
              "Retail Rate": 20.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kWh",
              "charge_network_by": "kWh",
              "has_peak_split": True,
              "afa_applicable": False,
              "ktwbb_applicable": True
            }
          },
          {
            "Tariff": "Medium Voltage General",
            "Voltage": "Medium Voltage",
            "Type": "General",
            "Rates": {
              "Energy Rate": 0.2983,
              "Capacity Rate": 29.43,
              "Network Rate": 59.84,
              "Retail Rate": 200.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kW",
              "charge_network_by": "kW",
              "has_peak_split": False,
              "afa_applicable": True,
              "ktwbb_applicable": True
            }
          },
          {
            "Tariff": "Medium Voltage TOU",
            "Voltage": "Medium Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.3132,
              "OffPeak Rate": 0.2723,
              "Capacity Rate": 30.19,
              "Network Rate": 66.87,
              "Retail Rate": 200.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kW (peak only)",
              "charge_network_by": "kW (peak only)",
              "has_peak_split": True,
              "afa_applicable": True,
              "ktwbb_applicable": True
            }
          },
          {
            "Tariff": "High Voltage General",
            "Voltage": "High Voltage",
            "Type": "General",
            "Rates": {
              "Energy Rate": 0.4303,
              "Capacity Rate": 16.68,
              "Network Rate": 14.53,
              "Retail Rate": 250.00
            },
            "Rules": {
              "charge_capacity_by": "kW",
              "charge_network_by": "kW",
              "has_peak_split": False,
              "afa_applicable": True,
              "ktwbb_applicable": False
            }
          },
          {
            "Tariff": "High Voltage TOU",
            "Voltage": "High Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.4452,
              "OffPeak Rate": 0.4043,
              "Capacity Rate": 21.76,
              "Network Rate": 23.06,
              "Retail Rate": 250.00
            },
            "Rules": {
              "charge_capacity_by": "kW (peak only)",
              "charge_network_by": "kW (peak only)",
              "has_peak_split": True,
              "afa_applicable": True,
              "ktwbb_applicable": False
            }
          }
        ]
      },
      "Specific Agriculture": { "Tariffs": [] },
      "Water & Sewerage Operator": { "Tariffs": [] },
      "Street Lighting": { "Tariffs": [] },
      "Co-Generation": { "Tariffs": [] },
      "Traction": { "Tariffs": [] },
      "Bulk": { 
        "Tariffs": [
          {
            "Tariff": "Bulk Medium Voltage TOU",
            "Voltage": "Medium Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.3032,
              "OffPeak Rate": 0.2623,
              "Capacity Rate": 30.21,
              "Network Rate": 37.17,
              "Retail Rate": 200.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kW (peak only)",
              "charge_network_by": "kW (peak only)",
              "has_peak_split": True,
              "afa_applicable": True,
              "ktwbb_applicable": True
            }
          },
          {
            "Tariff": "Bulk High Voltage TOU",
            "Voltage": "High Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.4052,
              "OffPeak Rate": 0.3643,
              "Capacity Rate": 29.24,
              "Network Rate": 26.02,
              "Retail Rate": 250.00
            },
            "Rules": {
              "charge_capacity_by": "kW (peak only)",
              "charge_network_by": "kW (peak only)",
              "has_peak_split": True,
              "afa_applicable": True,
              "ktwbb_applicable": False
            }
          }
        ]
      },
      "Thermal Energy Storage (TES)": { 
        "Tariffs": [
          {
            "Tariff": "TES Medium Voltage TOU",
            "Voltage": "Medium Voltage",
            "Type": "TOU",
            "Rates": {
              "Peak Rate": 0.3132,
              "OffPeak Rate": 0.2723,
              "Capacity Rate": 26.63,
              "Network Rate": 68.28,
              "Retail Rate": 200.00,
              "KTWBB Rate": 0.016
            },
            "Rules": {
              "charge_capacity_by": "kW (peak only)",
              "charge_network_by": "kW (peak only)",
              "has_peak_split": True,
              "afa_applicable": True,
              "ktwbb_applicable": True
            }
          }
        ]
      },
      "Backfeed": { "Tariffs": [] }
    }
  }
}

def get_tariff_data():
    return TARIFF_DATA