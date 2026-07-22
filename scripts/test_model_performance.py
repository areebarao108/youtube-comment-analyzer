import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# Set your remote tracking URI
mlflow.set_tracking_uri("http://3.111.55.134:5000")

@pytest.mark.parametrize("model_name, alias, holdout_data_path, vectorizer_path", [
    (
        "yt_chrome_plugin_model",
        "staging",
        "data/interim/test_processed.csv",
        "tfidf_vectorizer.pkl",
    ),
])
def test_model_performance(model_name, alias, holdout_data_path, vectorizer_path):
    try:
        # Load model using alias (MLflow 3.x)
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        # Load the holdout test data
        holdout_data = pd.read_csv(holdout_data_path)

        # Assuming first column contains text and last column contains labels
        X_holdout_raw = holdout_data.iloc[:, 0].fillna("")
        y_holdout = holdout_data.iloc[:, -1]

        # Apply TF-IDF transformation
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)

        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict
        y_pred = model.predict(X_holdout_tfidf_df)

        # Performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(
            y_holdout, y_pred,
            average="weighted",
            zero_division=1
        )
        recall = recall_score(
            y_holdout, y_pred,
            average="weighted",
            zero_division=1
        )
        f1 = f1_score(
            y_holdout, y_pred,
            average="weighted",
            zero_division=1
        )

        # Thresholds
        assert accuracy >= 0.40, f"Accuracy should be at least 0.40, got {accuracy:.4f}"
        assert precision >= 0.40, f"Precision should be at least 0.40, got {precision:.4f}"
        assert recall >= 0.40, f"Recall should be at least 0.40, got {recall:.4f}"
        assert f1 >= 0.40, f"F1 score should be at least 0.40, got {f1:.4f}"

        print(f"Performance test passed for '{model_name}@{alias}'")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")