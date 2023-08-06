"""train test script"""

from pathlib import Path

import pytest
import mlflow

from src.train import run_register_model


def test_run_register_model():
    """Test train script main function: run_register_model."""
    mlflow.set_tracking_uri(pytest.tracking_uri)
    run_register_model(data_path=pytest.processed_dir,
                       top_n=5,
                       download_registered_model=True,
                       target=pytest.target,
                       experiment_name=pytest.experiment_name,
                       hpo_experiment_name=pytest.hpo_experiment_name)

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{pytest.model_name}/{pytest.model_version}")
    model_path = "model/model.pkl"

    assert model.__class__.__name__ == "PyFuncModel", "The model was not loaded or another MLFlow artifact was loaded."
    assert (Path(__file__).parent.parent / model_path).is_file(), f"The model was not locally saved in: {model_path}"
