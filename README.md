# Customer Churn Prediction API

An end-to-end machine learning project: train a customer churn model, validate it properly, serve it behind a REST API, and containerize it for deployment.

Given a telecom customer's account details (contract type, tenure, monthly charges, services subscribed, etc.), the API returns the probability that the customer will churn, along with a risk tier that a retention team could act on.

## 🔗 Live Demo

[**Try the app →**](https://churn-predictor-ksxshqccqvbiu8ssegq7ej.streamlit.app/) (Streamlit UI — fill in a customer's details and get a live prediction)

[**API docs →**](https://churn-predictor-e6io.onrender.com/docs) (interactive Swagger docs for the underlying FastAPI backend)

*(Both run on free tiers — if they've been idle, the first load may take 30-60 seconds to wake up.)*

## Why this project

Most ML portfolio projects stop at a Jupyter notebook with an accuracy score. This one goes further:

* **Proper validation** — stratified train/test split, 5-fold cross-validation, and a held-out test set the model never touches during training or threshold tuning.
* **Realistic performance, not inflated numbers** — ROC-AUC of \~0.72. A dataset this noisy shouldn't produce 0.99 accuracy; if it does, something's leaking. This looks like real business data, not a toy example. See [`model/metrics.json`](model/metrics.json) for full metrics.
* **A tuned decision threshold** — instead of blindly using 0.5, the classification threshold is chosen to maximize F1 on the training set, since churn is imbalanced (\~33% positive class).
* **Served, not just modeled** — wrapped in a FastAPI service with input validation, batch prediction, and a health check, so it's actually deployable.
* **Containerized** — a Dockerfile so it can run anywhere (Render, Railway, ECS, Kubernetes) with one command.
* **Tested** — API tests verify the service works end-to-end, including a sanity check that the model ranks a loyal customer as lower-risk than a new, unsupported one.
* **A UI, not just an API** — a Streamlit front-end (`streamlit\\\_app/`) sits on top of the API so anyone can test predictions through a form, not just via `/docs` or `curl`.

## Architecture

```
data/generate\\\_data.py   -> synthetic but realistic telecom churn dataset (7,000 customers)
model/train.py          -> preprocessing + RandomForest pipeline, CV, evaluation, saves artifact
model/churn\\\_model.joblib -> trained model + metadata (feature order, threshold, train date)
app/main.py             -> FastAPI service that loads the artifact and serves predictions
streamlit\\\_app/app.py    -> Streamlit UI that calls the FastAPI backend
tests/test\\\_api.py       -> API test suite (pytest)
Dockerfile               -> container build for deployment
```

The preprocessing (scaling + one-hot encoding) and the model are bundled into a single `sklearn.Pipeline`, so the exact transform used at training time is guaranteed to be applied at inference time — no train/serve skew.

### Why a synthetic dataset?

The dataset is generated programmatically (`data/generate\\\_data.py`) rather than downloaded, with churn probability built from realistic risk factors (month-to-month contracts, short tenure, high charges, lack of tech support, electronic check payment, etc. all increase churn risk, matching patterns seen in real telecom churn data). This means:

* Anyone cloning the repo can reproduce it exactly — no Kaggle account or API key needed.
* The relationships between features and churn are known and interpretable, which makes it easy to sanity-check that the model actually learned something real (see feature importance below) rather than fitting noise.

## Results

|Metric|Value|
|-|-|
|Cross-val ROC-AUC (train, 5-fold)|0.712 ± 0.003|
|Test ROC-AUC|0.718|
|Test accuracy|0.67|
|Churn class recall|0.64|
|Churn class precision|0.50|

**On the precision/recall trade-off:** the classification threshold was chosen to maximize F1, but in a real retention use case, missing a churner (false negative) is usually more costly than flagging a loyal customer for an unnecessary retention offer (false positive) — since a missed churner is lost revenue, while an unnecessary offer just costs a discount. In production, this threshold would be tuned against an actual cost matrix from the business rather than optimized for F1 by default. The API exposes raw probabilities (`churn\\\_probability`) precisely so a downstream team can apply their own threshold instead of trusting a single cutoff baked into the model.

**Top predictive features:**

1. Monthly charges
2. Total charges
3. Tenure (months)
4. Month-to-month contract
5. Two-year contract (protective — lowers churn risk)
6. Payment via electronic check
7. Fiber optic internet
8. One-year contract
9. No tech support / no internet service

This matches domain intuition: new, high-paying, contract-free customers without support add-ons are the highest churn risk — exactly the profile a retention team would expect to target.

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python data/generate\\\_data.py

# 3. Train the model (produces model/churn\\\_model.joblib + model/metrics.json)
python model/train.py

# 4. Run the API
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for interactive Swagger docs, or test directly:

```bash
curl -X POST http://localhost:8000/predict \\\\
  -H "Content-Type: application/json" \\\\
  -d '{
    "gender": "Female", "senior\\\_citizen": 0, "partner": "Yes", "dependents": "No",
    "tenure\\\_months": 3, "phone\\\_service": "Yes", "multiple\\\_lines": "No",
    "internet\\\_service": "Fiber optic", "online\\\_security": "No", "online\\\_backup": "No",
    "device\\\_protection": "No", "tech\\\_support": "No", "streaming\\\_tv": "Yes",
    "streaming\\\_movies": "Yes", "contract": "Month-to-month", "paperless\\\_billing": "Yes",
    "payment\\\_method": "Electronic check", "monthly\\\_charges": 89.5, "total\\\_charges": 268.5
  }'
```

```json
{
  "churn\\\_probability": 0.7788,
  "churn\\\_prediction": "Yes",
  "risk\\\_tier": "High",
  "threshold\\\_used": 0.491
}
```

### Running the Streamlit UI locally

See [`streamlit\\\_app/README.md`](streamlit_app/README.md) for full instructions — in short:

```bash
cd streamlit\\\_app
pip install -r requirements.txt
streamlit run app.py
```

## Running with Docker

```bash
# Build (make sure the model is trained first, so churn\\\_model.joblib exists)
docker build -t churn-predictor .

# Run
docker run -p 8000:8000 churn-predictor
```

The API is then available at `http://localhost:8000`, identical to the local run above.

## Deploying

This container is ready to deploy as-is on any platform that builds from a Dockerfile:

* **Render / Railway** — connect the repo, they auto-detect the Dockerfile. *(This is how the live demo above is deployed — on Render's free tier.)*
* **AWS ECS / Google Cloud Run** — push the built image to a registry and deploy.

The `HEALTHCHECK` in the Dockerfile lets the platform know when the container is actually ready to serve traffic, not just started.

The Streamlit UI is deployed separately on **Streamlit Community Cloud**, pointed at the live Render API via an `API\\\_URL` secret — see [`streamlit\\\_app/README.md`](streamlit_app/README.md) for those steps.

## API Reference

|Endpoint|Method|Description|
|-|-|-|
|`/health`|GET|Liveness check|
|`/model-info`|GET|Model metadata (training date, threshold)|
|`/predict`|POST|Predict churn for a single customer|
|`/predict/batch`|POST|Predict churn for up to 500 customers in one request|

Full request/response schemas are auto-documented at `/docs` (Swagger UI) and `/redoc`.

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Tech stack

Python · scikit-learn · pandas · FastAPI · Pydantic · Docker · pytest · Streamlit

## Possible extensions

* Swap RandomForest for XGBoost/LightGBM and compare via the same CV harness
* Add SHAP values to the `/predict` response for per-customer explainability
* Add a `/retrain` endpoint or a scheduled retraining job as new data arrives
* Track model versions and metrics over time (e.g., with MLflow)
* Add authentication and request logging for production use



