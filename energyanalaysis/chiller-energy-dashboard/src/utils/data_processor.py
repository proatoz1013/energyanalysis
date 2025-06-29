def load_data(file_path):
    """Load data from an Excel or CSV file."""
    import pandas as pd
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

def detect_numeric_columns(df):
    """Detect numeric columns in the DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()

def prepare_data(df, time_col, power_col, flowrate_col, cooling_load_col, efficiency_col):
    """Prepare the data for analysis by selecting relevant columns."""
    required_columns = [time_col, power_col, flowrate_col, cooling_load_col, efficiency_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("One or more specified columns are not in the DataFrame.")
    
    return df[required_columns]

def calculate_power_usage(df, power_col):
    """Calculate total power usage from the specified power column."""
    return df[power_col].sum()

def calculate_efficiency(df, cooling_load_col, power_col):
    """Calculate efficiency metrics based on cooling load and power usage."""
    return df[cooling_load_col] / df[power_col]

def calculate_performance_metrics(df, power_col, cooling_load_col):
    """Calculate performance metrics such as kW/TR and COP."""
    total_power = calculate_power_usage(df, power_col)
    total_cooling_load = df[cooling_load_col].sum()
    
    kW_TR = total_power / total_cooling_load if total_cooling_load > 0 else 0
    COP = total_cooling_load / total_power if total_power > 0 else 0
    
    return {
        'total_power': total_power,
        'kW_TR': kW_TR,
        'COP': COP
    }