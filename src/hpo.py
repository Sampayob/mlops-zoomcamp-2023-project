"""hpo script"""

import configparser
from pathlib import Path
import logging

import matplotlib
import mlflow
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils.utils import load_pickle

matplotlib.use('Agg')  # to avoid mlflow matplotlib backend error
logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "config.ini"))


HPO_EXPERIMENT_NAME = str(config["mlflow"]["hpo_experiment_name"])
TARGET = str(config["train"]["target_name"])
PROCESSED_DIR = str(config["paths"]["dir_processed_data"])
N_TRIALS = int(str(Path(config["hpo"]["num_trials"])))


def run_optimization(data_path: str = PROCESSED_DIR, num_trials: int = N_TRIALS, target: str = TARGET) -> None:
    """Train multiple Random Forest classification modelsusing hyperparameter optimization
    with optuna and log them to MLFlow.
    """
    mlflow.sklearn.autolog()

    mlflow.set_tracking_uri(str(config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)

    logger.info("Load data splits.")
    train = load_pickle(str(Path(__file__).parent.parent / data_path / "train.pkl"))
    val = load_pickle(str(Path(__file__).parent.parent / data_path / "val.pkl"))
    X_train, y_train = train.drop([target], axis=1), train[target]
    X_val, y_val = val.drop([target], axis=1), val[target]
    del train, val

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 100, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, 1),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="micro")
            mlflow.log_metric("F1", f1)

        return f1

    logger.info("Start optuna study.")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)
    logger.info("Finish optuna study.")


if __name__ == "__main__":
    run_optimization()
