"""train script"""

import configparser
from pathlib import Path
import logging

import matplotlib
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils.utils import load_pickle


matplotlib.use('Agg')  # to avoid mlflow matplotlib backend error
logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "config.ini"))

HPO_EXPERIMENT_NAME = str(config["mlflow"]["hpo_experiment_name"])
EXPERIMENT_NAME = str(config["mlflow"]["register_model_experiment_name"])
RF_PARAMS = str(config["train"]["params"]).split(',')
PROCESSED_DIR = str(config["paths"]["dir_processed_data"])
TARGET = str(config["train"]["target_name"])
TOP_N_RUNS = int(str(Path(config["train"]["top_n_runs"])))


def train_and_log_model(params: dict, splits_dict: dict) -> None:
    """Train Random Forest model and log metrics to MLFlow."""
    final_params = {}

    with mlflow.start_run():
        for param in RF_PARAMS:
            param = param.strip()
            final_params[param] = int(params[param])

        rf = RandomForestClassifier(**final_params)
        rf.fit(splits_dict['X_train'], splits_dict['y_train'])

        # Evaluate model on the validation and test sets
        val_f1 = f1_score(splits_dict['y_val'], rf.predict(splits_dict['X_val']), average='micro')
        mlflow.log_metric("val_f1", val_f1)
        test_f1 = f1_score(splits_dict['y_test'], rf.predict(splits_dict['X_test']), average='micro')
        mlflow.log_metric("test_f1", test_f1)


def run_register_model(data_path: str = PROCESSED_DIR,
                       top_n: int = TOP_N_RUNS,
                       download_registered_model=True,
                       target: str = TARGET,
                       experiment_name: str = EXPERIMENT_NAME,
                       hpo_experiment_name: str = HPO_EXPERIMENT_NAME) -> None:
    """Train and log best HPO models, select the best one and register/download it."""

    client = MlflowClient()

    mlflow.set_tracking_uri(str(config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    # Load data splits
    logger.info("Load data splits.")
    train = load_pickle(str(Path(__file__).parent.parent / data_path / "train.pkl"))
    val = load_pickle(str(Path(__file__).parent.parent / data_path / "val.pkl"))
    test = load_pickle(str(Path(__file__).parent.parent / data_path / "test.pkl"))
    X_train, y_train = train.drop([target], axis=1), train[target]
    X_val, y_val = val.drop([target], axis=1), val[target]
    X_test, y_test = test.drop([target], axis=1), test[target]
    del train, val, test

    splits_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    # Retrieve the top_n model runs and log the models
    logger.info("Log models from best HPO runs.")
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.f1 DESC"]
    )

    for run in runs:
        train_and_log_model(params=run.data.params, splits_dict=splits_dict)

    # Select the model with the highest test F1
    logger.info("Select best model.")
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_f1 DESC"])[0]

    # Register the best model
    logger.info("Register best model.")
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name=str(config["mlflow"]["register_model_name"])
    )

    # Download best model locally
    if download_registered_model:
        client.download_artifacts(run_id=best_run.info.run_id,
                                  path="model/model.pkl",
                                  dst_path=str(Path(__file__).parent.parent))


if __name__ == '__main__':
    run_register_model()
