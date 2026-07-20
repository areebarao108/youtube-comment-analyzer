import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://3.111.55.134:5000")


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        client = mlflow.tracking.MlflowClient()

        # Get the run to find its experiment_id
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id

        # Find the Logged Model tied to this run
        logged_models = mlflow.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id='{run_id}'",
            output_format="list"
        )

        if not logged_models:
            raise ValueError(f"No logged model found for run_id {run_id}")

        logged_model_id = logged_models[0].model_id
        model_uri = f"models:/{logged_model_id}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Set an alias instead of the deprecated stage transition
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered and aliased as "staging".')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()