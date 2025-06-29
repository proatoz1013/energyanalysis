import pytest
import pandas as pd
from src.utils.data_processor import process_data

def test_process_data_valid_csv():
    df = process_data('data/sample_data/sample_chiller_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Chiller Power' in df.columns
    assert 'Cooling Load' in df.columns

def test_process_data_invalid_file():
    with pytest.raises(ValueError):
        process_data('data/uploads/invalid_file.txt')

def test_process_data_missing_columns():
    df = process_data('data/sample_data/sample_chiller_data.csv')
    df_missing = df.drop(columns=['Chiller Power'])
    with pytest.raises(KeyError):
        process_data(df_missing)

def test_process_data_excel_file():
    df = process_data('data/sample_data/sample_chiller_data.xlsx')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Chiller Power' in df.columns
    assert 'Cooling Load' in df.columns

def test_process_data_numeric_columns():
    df = process_data('data/sample_data/sample_chiller_data.csv')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    assert 'Chiller Power' in numeric_cols
    assert 'Cooling Load' in numeric_cols