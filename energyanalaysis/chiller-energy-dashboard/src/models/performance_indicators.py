class PerformanceIndicators:
    def __init__(self):
        self.total_power_usage = 0
        self.kW_per_TR = 0
        self.coefficient_of_performance = 0
        self.average_efficiency = 0

    def calculate_total_power_usage(self, power_data):
        self.total_power_usage = sum(power_data)

    def calculate_kw_per_tr(self, power_data, cooling_load_data):
        if cooling_load_data:
            self.kW_per_TR = sum(power_data) / sum(cooling_load_data)

    def calculate_cop(self, power_data, cooling_load_data):
        if cooling_load_data:
            self.coefficient_of_performance = sum(cooling_load_data) / sum(power_data)

    def calculate_average_efficiency(self, efficiency_data):
        if efficiency_data:
            self.average_efficiency = sum(efficiency_data) / len(efficiency_data)

    def get_performance_metrics(self):
        return {
            "Total Power Usage": self.total_power_usage,
            "kW/TR": self.kW_per_TR,
            "COP": self.coefficient_of_performance,
            "Average Efficiency": self.average_efficiency,
        }