"""orchestrate script"""

from pathlib import Path
import logging
import configparser

from prefect import flow, task
import matplotlib

from transform import run_transform
from hpo import run_optimization
from train import run_register_model

matplotlib.use('Agg')  # to avoid mlflow matplotlib backend error
logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "config.ini"))


PROCESSED_DIR = str(Path(config["paths"]["dir_processed_data"]))
RAW_DIR = str(Path(config["paths"]["raw_data"]))
N_TRIALS = int(str(Path(config["hpo"]["num_trials"])))
TOP_N_RUNS = int(str(Path(config["train"]["top_n_runs"])))
PREFECT_FLOW_NAME = str(config["orchestration"]["prefect_experiment_name"])
TARGET = str(config["train"]["target_name"])
HPO_EXPERIMENT_NAME = str(config["mlflow"]["hpo_experiment_name"])
EXPERIMENT_NAME = str(config["mlflow"]["register_model_experiment_name"])


@task(retries=3, retry_delay_seconds=2)
def read_and_transform_data() -> None:
    """Transform raw data into processed data splits: train, val test and save the splits."""
    run_transform(data_path=RAW_DIR)


@task(retries=3, retry_delay_seconds=2)
def hpo() -> None:
    """Run HPO with Optuna to get the best model."""
    run_optimization(data_path=PROCESSED_DIR, num_trials=5, target=TARGET)


@task(retries=3, retry_delay_seconds=2, log_prints=True)
def train() -> None:
    """Retrieve top_n runs, train and log models, select best one and register it in mlflow."""
    run_register_model(data_path=PROCESSED_DIR,
                       top_n=TOP_N_RUNS,
                       download_registered_model=True,
                       experiment_name=EXPERIMENT_NAME,
                       hpo_experiment_name=HPO_EXPERIMENT_NAME)


@flow(name=PREFECT_FLOW_NAME)
def main_flow() -> None:
    """The main training pipeline"""

    # Load and Transform
    read_and_transform_data()

    # HPO
    hpo()

    # Train
    train()


if __name__ == "__main__":
    main_flow()
