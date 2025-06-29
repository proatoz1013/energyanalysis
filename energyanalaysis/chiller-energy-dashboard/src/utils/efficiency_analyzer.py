from typing import Dict, Any
import pandas as pd

def calculate_efficiency(data: pd.DataFrame, power_col: str, cooling_load_col: str) -> pd.Series:
    """
    Calculate the efficiency metrics based on power usage and cooling load.

    Args:
        data (pd.DataFrame): The DataFrame containing chiller data.
        power_col (str): The name of the column representing chiller power usage.
        cooling_load_col (str): The name of the column representing cooling load.

    Returns:
        pd.Series: A Series containing the calculated efficiency metrics.
    """
    efficiency = data[cooling_load_col] / data[power_col]
    return efficiency

def analyze_efficiency(data: pd.DataFrame, power_col: str, cooling_load_col: str) -> Dict[str, Any]:
    """
    Analyze the efficiency of the chiller plant.

    Args:
        data (pd.DataFrame): The DataFrame containing chiller data.
        power_col (str): The name of the column representing chiller power usage.
        cooling_load_col (str): The name of the column representing cooling load.

    Returns:
        Dict[str, Any]: A dictionary containing various efficiency metrics.
    """
    efficiency = calculate_efficiency(data, power_col, cooling_load_col)
    
    analysis_results = {
        'average_efficiency': efficiency.mean(),
        'max_efficiency': efficiency.max(),
        'min_efficiency': efficiency.min(),
        'efficiency_std_dev': efficiency.std(),
        'efficiency_series': efficiency
    }
    
    return analysis_results