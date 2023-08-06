import configparser
from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame


config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "src" / "config.ini"))

pytest.raw_dir = str(config["paths"]["raw_data"])
pytest.train_path = str(Path(config["paths"]["train_data"]))
pytest.val_path = str(Path(config["paths"]["val_data"]))
pytest.test_path = str(Path(config["paths"]["test_data"]))
pytest.processed_dir = str(config["paths"]["dir_processed_data"])
pytest.n_trials = int(str(Path(config["hpo"]["num_trials"])))
pytest.hpo_experiment_name = str(config["mlflow"]["hpo_experiment_name"])
pytest.target = str(config["train"]["target_name"])
pytest.tracking_uri = str(config["mlflow"]["tracking_uri"])
pytest.model_name = "dermatology-disease-random-forest"
pytest.model_version = "1"
pytest.experiment_name = str(config["mlflow"]["register_model_experiment_name"])


@pytest.fixture(scope="module")
def input_transform_data() -> DataFrame:
    data = [
        [1, 33, 2, 3],
        [0, '?', 1, 0],
        [3, 48, 2, 4],
        [2, 45, 1, 3],
    ]

    columns = [
        "class",
        "age",
        "erythema",
        "definite_borders",
    ]

    df = pd.DataFrame(data, columns=columns)
    df['class'] = df['class'].astype(float)
    df['age'] = df['age'].astype('str')
    df['erythema'] = df['erythema'].astype(float)
    df['definite_borders'] = df['definite_borders'].astype(int)

    return df
