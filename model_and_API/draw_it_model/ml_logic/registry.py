import mlflow
from mlflow.tracking import MlflowClient
import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow.keras import Model, models


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    Save trained model, params and metrics
    """
    #Get timestamp to use for model naming
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #Save model to mlflow, if defined
    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = os.environ["MLFLOW_EXPERIMENT"]
        model_name = os.environ["MLFLOW_MODEL_NAME"]

        # configure mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            if params is not None:
                mlflow.log_params(params)

            # STEP 2: push metrics to mlflow

            if metrics is not None:
                mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            if model is not None:
               mlflow.keras.log_model(keras_model=model,
                    artifact_path="model",
                    keras_module="tensorflow.keras",
                    registered_model_name=model_name)

            print("\n✅ data saved in mlflow")

        return None

    #Save model locally
    if os.environ.get("MODEL_TARGET") == "local":

        #Path for storage
        local_registry_path = os.environ["LOCAL_REGISTRY_PATH"]

        # save params
        if params is not None:
            params_path = os.path.join(local_registry_path, "params", timestamp + ".pickle")
            print(f"- params path: {params_path}")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)

        # save metrics
        if metrics is not None:
            metrics_path = os.path.join(local_registry_path, "metrics", timestamp + ".pickle")
            print(f"- metrics path: {metrics_path}")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)

        # save model
        if model is not None:
            model_path = os.path.join(local_registry_path, "models", timestamp)
            print(f"- model path: {model_path}")
            model.save(model_path)

        print("\n✅ data saved locally")

    return None


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """

    #Load model from mlflow, if defined
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = "Production"

        print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)


        mlflow.set_tracking_uri("https://mlflow.lewagon.ai")
        model_name = os.environ["MLFLOW_MODEL_NAME"]
        model_uri = f"models:/{model_name}/Production"

        model = mlflow.keras.load_model(model_uri=model_uri)


        return model


    #Load model from locally (default)
    print(Fore.BLUE + "\nLoad model locally..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), "models")
    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None
    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    #Load model
    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
