# YouTube Comment Analyzer — NLP Backend

An end-to-end MLOps pipeline for sentiment analysis of YouTube comments.
Trains and tracks NLP models with MLflow, versions data and pipelines with DVC,
and serves predictions via a containerized Flask API deployed on AWS, integrated into a Chrome extension.

## What It Does

- Fetches YouTube comments via a Chrome extension frontend
- Performs sentiment/toxicity classification using an optimized LightGBM model
- Serves real-time predictions through a REST API
- Full MLOps lifecycle: tracked experiments → reproducible pipelines → containerized API → CI/CD → AWS deployment

## Tech Stack

| Layer | Tools |
|---|---|
| ML & NLP | Python, Scikit-learn, LightGBM, TF-IDF |
| MLOps | MLflow (experiment tracking), DVC (data/pipeline versioning), S3 (artifact storage) |
| Backend | Flask, REST API |
| Frontend | Chrome Extension (JavaScript, HTML, CSS) |
| DevOps | Docker, GitHub Actions CI/CD, AWS ECR, AWS CodeDeploy |

## Model Development Pipeline

Trained on labeled Reddit sentiment dataset with rigorous experiment tracking:

| Experiment | Focus | Key Decision |
|---|---|---|
| Exp 1 | Vectorization | TF-IDF outperformed Bag-of-Words |
| Exp 2 | Feature selection | Tuned max_features for optimal representation |
| Exp 3 | Class imbalance | ADASYN yielded best class distribution |
| Exp 4 | Model selection | LightGBM outperformed Random Forest and others |
| Exp 5 | Hyperparameter tuning | Final model: **86% accuracy** |

All experiments logged to MLflow with metrics, parameters, and confusion matrices.

## DVC Pipeline

Reproducible pipeline with 5 stages:

```yaml
dvc.yaml
├── data_ingestion      # Fetch and store raw data to S3
├── preprocessing       # Clean, tokenize, TF-IDF vectorization
├── model_building      # Train LightGBM with tracked hyperparameters
├── evaluation          # Compute accuracy, precision, recall, confusion matrix
└── model_registry      # Register best model to MLflow model registry
```


DVC tracks data and pipeline state. Dataset artifacts stored in AWS S3.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Batch sentiment prediction for YouTube comments |
| `/predict_with_timestamps` | POST | Sentiment prediction with comment timestamps |
| `/generate_chart` | POST | Pie chart visualization of sentiment distribution |
| `/generate_wordcloud` | POST | Word cloud from preprocessed comment text |
| `/generate_trend_graph` | POST | Monthly sentiment trend graph over time |

## Chrome Extension Integration
YouTube Page → Chrome Extension (JS) → Flask API (Docker on AWS)→ MLflow Model Registry
DVC + S3 Artifacts


Frontend repo: [github.com/areebarao108/yt-chrome-plugin-frontend](https://github.com/areebarao108/yt-chrome-plugin-frontend)

## CI/CD & Deployment

GitHub Actions workflow automates:

1. Run DVC pipeline end-to-end
2. Model loading validation test
3. Model signature test
4. Performance test against accuracy threshold
5. Promote passing model to "Production" stage in MLflow
6. Build and start Flask API
7. Run API integration tests
8. Build Docker image and push to **AWS ECR**
9. Deploy to **AWS CodeDeploy** for live production serving

## How to Run Locally

```bash
# Clone
git clone https://github.com/areebarao108/yt_comment_analyzer.git
cd yt_comment_analyzer

# Install dependencies
pip install -r requirements.txt

# Run DVC pipeline
dvc repro

# Start Flask API
python app.py
```
Run with Docker
docker build -t yt-chrome-plugin .
docker run -p 5000:5000 yt-chrome-plugin

##Project Structure
```
yt_comment_analyzer/
├── app.py                    # Flask API (5 endpoints)
├── Dockerfile                # Container config
├── requirements.txt          # Python dependencies
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # Pipeline reproducibility lock
├── .github/workflows/        # CI/CD configuration
├── src/                      # Pipeline stage scripts
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model_building.py
│   ├── evaluation.py
│   └── model_registry.py
├── notebooks/                # EDA and baseline experiments
├── models/                   # Serialized model artifacts
└── README.md
```

Why Not BERT?
BERT and transformer-based models would likely improve performance further.
However, this project intentionally uses classical ML (LightGBM + TF-IDF) to demonstrate
end-to-end MLOps engineering, pipeline reproducibility, and production deployment—skills
that transfer directly to deep learning systems.

