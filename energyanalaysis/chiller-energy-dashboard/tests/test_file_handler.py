import pytest
import pandas as pd
from src.utils.file_handler import read_file, write_file

def test_read_csv_file():
    df = read_file('data/sample_data/sample_chiller_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_read_excel_file():
    df = read_file('data/sample_data/sample_chiller_data.xlsx')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_write_file(tmp_path):
    data = {'Chiller Power': [100, 200, 300], 'Cooling Load': [80, 160, 240]}
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_output.csv"
    write_file(df, file_path)
    
    # Verify the file was created and contains the correct data
    assert file_path.exists()
    read_back_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(df, read_back_df)

def test_invalid_file_format():
    with pytest.raises(ValueError):
        read_file('data/sample_data/invalid_file.txt')