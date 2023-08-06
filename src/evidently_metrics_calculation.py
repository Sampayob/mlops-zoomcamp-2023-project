"""evidently metrics script"""

import datetime
import logging
import random
import time
from pathlib import Path
import configparser

import matplotlib
import pandas as pd
import psycopg
import mlflow
from evidently import ColumnMapping
from evidently.metrics import (ColumnDriftMetric,
                               ConflictPredictionMetric,
                               DatasetDriftMetric,
                               DatasetMissingValuesMetric)
from evidently.report import Report
from prefect import flow, task

from utils.utils import load_pickle

matplotlib.use('Agg')  # to avoid mlflow matplotlib backend error
logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(str(Path(__file__).parent / "config.ini"))

PROCESSED_DIR = str(config["paths"]["dir_processed_data"])
MODEL_NAME = str(config["monitoring"]["model_name"])
MODEL_VERSION = str(config["monitoring"]["model_version"])
DB_NAME = str(config["monitoring"]["db_name"])
TABLE_NAME = str(config["monitoring"]["table_name"])
DB_PASSWORD = str(config["monitoring"]["db_password"])
DB_USER = str(config["monitoring"]["db_user"])
DB_HOST = str(config["monitoring"]["db_localhost"])
DB_PORT = str(config["monitoring"]["db_port"])
SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists {0};
create table {1}(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
 	num_not_stable_prediction integer
)
""".format(TABLE_NAME, TABLE_NAME)

today_date = datetime.datetime.now()
begin = datetime.datetime(today_date.year, today_date.month, today_date.day, 0, 0)
reference_data = load_pickle(str(Path(__file__).parent.parent / PROCESSED_DIR / "train.pkl"))
current_data = load_pickle(str(Path(__file__).parent.parent / PROCESSED_DIR / "test.pkl"))

mlflow.set_tracking_uri(str(config["mlflow"]["tracking_uri"]))
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")

num_features = reference_data.drop(['class'], axis=1).columns.tolist()
column_mapping = ColumnMapping(
    prediction='class',
    numerical_features=num_features,
    categorical_features=None,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name="class"),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ConflictPredictionMetric(),
]
)

@task(retries=2, retry_delay_seconds=5, name="DB: preparation")
def prep_db():
	with psycopg.connect(f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASSWORD}", autocommit=True) as conn:
		res = conn.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'")
		if len(res.fetchall()) == 0:
			conn.execute(f"create database {DB_NAME};")
		with psycopg.connect(f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}") as conn:
			conn.execute(create_table_statement)

@task(retries=2, retry_delay_seconds=5, name="DB: calculate metrics")
def calculate_metrics_postgresql(curr, i):

	current_data['class'] = model.predict(current_data[num_features].fillna(0))

	report.run(reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
	num_not_stable_prediction = result['metrics'][3]['result']['current']['number_not_stable_prediction']

	curr.execute(
		f"insert into {TABLE_NAME}" \
        "(timestamp, prediction_drift, num_drifted_columns, share_missing_values, num_not_stable_prediction)" \
        "values (%s, %s, %s, %s, %s)",
		(begin + datetime.timedelta(i),
      	 prediction_drift,
         num_drifted_columns,
         share_missing_values,
         num_not_stable_prediction)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect(f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
