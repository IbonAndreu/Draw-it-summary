# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:2.9.1
WORKDIR /
COPY draw_it_model draw_it_model
# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements_light.txt requirements.txt
COPY setup.py setup.py

# Copy .env
COPY .env .env
COPY .envrc .envrc

# A build time, download the model from the MLflow server and copy it once for all inside of the image
COPY training_output/models training_output/models
RUN pip install --upgrade pip
RUN pip install .

# Start model
CMD uvicorn draw_it_model.api.fast:app --host 0.0.0.0 --port $PORT
