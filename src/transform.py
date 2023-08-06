"""transform script"""

from pathlib import Path
import configparser
import logging
from typing import Tuple

from pandas import DataFrame
import pandas as pd

from utils.utils import load_pickle, train_val_test_split, save_data_splits


logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "config.ini"))

TARGET = str(config["train"]["target_name"])
RAW_DIR = str(config["paths"]["raw_data"])
PROCESSED_DIR = str(Path(config["paths"]["dir_processed_data"]))
TRAIN_PATH = str(Path(config["paths"]["train_data"]))
VAL_PATH = str(Path(config["paths"]["val_data"]))
TEST_PATH = str(Path(config["paths"]["test_data"]))


def preprocess_data(data: DataFrame) -> DataFrame:
    """Prepare data before data splits."""
    # Remove symbols 
    data = data[data['age'] != '?']
    # Convert features dtype integer to float to hanlde possible/future missing values
    data = data.astype("float64")
    # target class as integer
    data['class'] = data['class'].astype("int16")

    return data


def run_transform(data_path: str = RAW_DIR) -> None:
    logger.info("Load raw data: %s.", data_path)
    data = pd.read_csv(Path(__file__).parent.parent / data_path)

    logger.info("Preprocess raw data.")
    data = preprocess_data(data)

    logger.info("Create data splits: train, val, test.")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data.drop([TARGET], axis=1), data[TARGET])

    logger.info("Save data splits: %s.", PROCESSED_DIR)
    (Path(__file__).parent.parent / PROCESSED_DIR).mkdir(exist_ok=True, parents=True)

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    save_data_splits(
        train,
        val,
        test,
        str(Path(__file__).parent.parent / TRAIN_PATH),
        str(Path(__file__).parent.parent / VAL_PATH),
        str(Path(__file__).parent.parent / TEST_PATH)
    )


if __name__ == '__main__':
    run_transform()
