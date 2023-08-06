import pickle
from typing import Tuple

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


def load_pickle(filename):
    """Return .pkl/.pickle file from filename path."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_val_test_split(X, y) -> Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """Split data in train, validation and test splits."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data_splits(train: DataFrame, val: DataFrame, test: DataFrame, train_path: str, val_path: str, test_path: str) -> None:
    """Save input data splits into .pkl files."""
    for path, file in [[train_path, train],
                       [val_path, val],
                       [test_path, test]]:

        file.to_pickle(path)
