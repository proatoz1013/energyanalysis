# Old TNB charging rates for reference and comparison

# Mapping of old tariffs to their closest equivalent new tariffs
old_to_new_tariff_map = {
    "E1 - Medium Voltage General": "Medium Voltage General",
    "E2 - Medium Voltage Peak/Off-Peak": "Medium Voltage TOU",
    "E3 - High Voltage Peak/Off-Peak": "High Voltage TOU",
    "D - Low Voltage Industrial": "Low Voltage General",
    "C1 - Medium Voltage Commercial": "Medium Voltage General / TOU",
    "C2 - Medium Voltage Commercial": "Medium Voltage General",
    "D - Domestic Tariff": "Domestic Tariff"
}

charging_rates = {
    "E1 - Medium Voltage General": "Base: RM 0.337/kWh, MD: RM 29.60/kW",
    "E2 - Medium Voltage Peak/Off-Peak": "Peak: RM 0.355/kWh, Off-Peak: RM 0.219/kWh, MD: RM 37.00/kW",
    "E3 - High Voltage Peak/Off-Peak": "Peak: RM 0.337/kWh, Off-Peak: RM 0.202/kWh, MD: RM 35.50/kW",
    "D - Low Voltage Industrial": "Tiered: RM 0.38/kWh (1-200 kWh), RM 0.441/kWh (>200 kWh)",
    "C1 - Medium Voltage Commercial": "Flat: RM 0.365/kWh, MD: RM 30.3/kW, ICPT: RM 0.16/kWh",
    "C2 - Medium Voltage Commercial": "Peak: RM 0.365/kWh, Off-Peak: RM 0.224/kWh, MD: RM 45.1/kW, ICPT: RM 0.16/kWh",
    "D - Domestic Tariff": "Tiered: RM 0.218–0.571/kWh, ICPT: -RM 0.02 to RM 0.10/kWh depending on usage tier"
}
