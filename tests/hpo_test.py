"""hpo test script"""

import pytest
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.hpo import run_optimization


def test_run_optimization():
    """Test hpo script main function: run_optimization."""
    run_optimization(pytest.processed_dir, pytest.n_trials, pytest.target)

    client = MlflowClient()

    mlflow.set_tracking_uri(pytest.tracking_uri)

    experiment = client.get_experiment_by_name(pytest.hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.f1 DESC"]
    )

    assert len(runs) > 0, f"There is not runs in experiment {pytest.hpo_experiment_name}"
