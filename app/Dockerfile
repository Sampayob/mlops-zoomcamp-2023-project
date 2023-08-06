FROM python:3.10.11-slim

ENV TRACKING_URI=http://0.0.0.0:5000
ENV MODEL_NAME=dermatology-disease-random-forest
ENV MODEL_VERSION=1

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

