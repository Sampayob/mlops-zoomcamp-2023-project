"""transform test script"""

from pathlib import Path
import pytest

from src.transform import preprocess_data, run_transform


def test_preprocess_data(input_transform_data):
    """Test transform script preprocess_data function."""
    result = preprocess_data(input_transform_data)

    assert len(result[result["age"].astype(str) == '?']) == 0, "Symbols has been not removed from 'age' feature."
    assert result.dtypes.nunique() == 2, "There is more or less dtypes than the desired ones."
    assert result.dtypes.unique().tolist()[0] in ['float64', "int16"], "There different dtypes than the desired ones."
    assert result.dtypes.unique().tolist()[1] in ['float64', "int16"], "There different dtypes than the desired ones."


def test_run_transform():
    """Test transform script main function: run_transform."""
    run_transform(pytest.raw_dir)
    assert Path(pytest.train_path).is_file()
    assert Path(pytest.train_path).is_file()
    assert Path(pytest.train_path).is_file()
