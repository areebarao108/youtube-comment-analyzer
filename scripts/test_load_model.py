import mlflow
import mlflow.pyfunc
import pytest

mlflow.set_tracking_uri("http://3.111.55.134:5000")

@pytest.mark.parametrize("model_name, alias", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_latest_model(model_name, alias):

    model_uri = f"models:/{model_name}@{alias}"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        assert model is not None
        print(f"{model_name}@{alias} loaded successfully")

    except Exception as e:
        pytest.fail(str(e))