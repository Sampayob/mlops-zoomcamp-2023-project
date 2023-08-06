"""prediction script"""

import os
import pickle
from pathlib import Path
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request
from requests.exceptions import ConnectionError


logging.basicConfig(level='INFO', format="%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_VERSION = os.getenv('MODEL_VERSION')


def load_model():
    mlflow.set_tracking_uri(os.getenv('TRACKING_URI'))
    return mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")


def prepare_data(data):
    logger.debug('Prepare inference data')
    data = pd.DataFrame(data, index=[0])
    data = data.astype("float64")
    return data


def predict(features):
    logger.debug('Predict')
    model = load_model()
    return int(model.predict(features)[0])


app = Flask('dermatologydisease-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    inference = request.get_json()
    features = prepare_data(inference)
    pred = predict(features)
    logger.debug('Post result')
    result = {
        'predicted disease': pred,
        'model': {'name': MODEL_NAME, 'version': MODEL_VERSION}
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
