import mlflow


def promote_model():
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri("http://3.111.55.134:5000")

    client = mlflow.MlflowClient()

    model_name = "yt_chrome_plugin_model"

    try:
        # Get the version that has the "staging" alias
        staging_model = client.get_model_version_by_alias(
            model_name,
            "staging"
        )

        staging_version = staging_model.version

        # Move the "production" alias to this version
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=staging_version
        )

        print(
            f"Model version {staging_version} promoted to Production successfully."
        )

    except Exception as e:
        print(f"Error promoting model: {e}")


if __name__ == "__main__":
    promote_model()