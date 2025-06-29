import os
import pandas as pd

def read_file(file_path):
    """Read an Excel or CSV file and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

def save_file(dataframe, file_path):
    """Save a DataFrame to a CSV file."""
    dataframe.to_csv(file_path, index=False)

def get_numeric_columns(dataframe):
    """Return a list of numeric columns in the DataFrame."""
    return dataframe.select_dtypes(include=['number']).columns.tolist()