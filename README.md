# MLOps Zoomcamp 2023 Project

## Project description

This project aims to put in practice the learnings from the [mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by [DataTalksClub](https://github.com/DataTalksClub).

It consist in applying mlops practices when training and serving a machine-learning model.

In this case, this is a supervised multi-class classifier trained on the [dermatology Dataset](https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification) to identify "erythemato-squamous" diseases:

- **Dataset description**:

    - The disseases share the clinical features of erythema and scaling, with minimal differences.
    - The disorders in this group are psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris.
    - Usually, a biopsy is necessary for the diagnosis, but unfortunately, these diseases share many histopathological features as well.

- **Features information**:
    - Patients were first evaluated clinically with 12 features. Afterward, skin samples were taken for the evaluation of 22 histopathological features.
    - The values of the histopathological features are determined by an analysis of the samples under a microscope.
    - In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise.
    - Every other feature clinical and histopathological was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.

## Instructions
### Pipeline/Orchestration (Experiment tracking and model registry)
1. Launch `mlflow ui`: it can be launched with the preferred host:
```bash
mlflow ui  # --host 0.0.0.0
```
2. Launch `prefect server`:
```bash
prefect server start
```
3. Launch `orchestrate.py` script
```bash
python orchestrate.py
```

### Deployment
1. Check out the ENV variables listed in the `Dockerfile` or overwrite then with your own or set them from the command line (`export` command):
```bash
export NAME=VALUE
```

2. Launch `mlflow server` where your database and/or your mlruns/mlartifacts directories are (previously generated running orchestrate.py or the transform.py, hpo.py, train.py scripts individually):
```bash 
mlflow server --host 0.0.0.0
```

3. Build docker image from `/app`:
```bash 
docker build -t dermatology-disease-prediction-service .
```

4. Run docker container:
```bash
docker run -it --rm -p 9696:9696 --network="host" dermatology-disease-prediction-service
```

5. Test the app:
```bash
python test.py
```

### Monitoring

1. From `/config` launch `docker-compose` to build a database and Grafana:
```bash
 docker-compose up --build
 ```
 
2. Feed data to the database:
```bash
 python src/evidently_metrics_calculation.py
 ```
3. Enter [localhost:3000](https://www.localhost:3000) to enter to Grafana dashboard (user: admin, password: admin (first time))
   
4. Enter to see the dashboards. 