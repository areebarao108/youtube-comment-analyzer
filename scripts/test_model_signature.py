import mlflow
import mlflow.pyfunc
import pytest
import pandas as pd
import pickle

mlflow.set_tracking_uri("http://3.111.55.134:5000")

@pytest.mark.parametrize("model_name, alias, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, alias, vectorizer_path):

    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        input_text = "hi how are you"

        input_data = vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        prediction = model.predict(input_df)

        assert input_df.shape[1] == len(vectorizer.get_feature_names_out())
        assert len(prediction) == 1

        print("Model loaded and prediction successful.")

    except Exception as e:
        pytest.fail(f"Model test failed: {e}")