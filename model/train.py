"""
Trains a customer churn prediction model.

Approach:
- Preprocessing (encoding, scaling) is bundled into a single sklearn Pipeline
  with the model, so the exact same transform used in training is applied
  at inference time -- no train/serve skew.
- Stratified train/test split (churn is imbalanced ~33/67).
- 5-fold cross-validation on the training set to get a stable estimate of
  performance before touching the test set.
- Final evaluation on a held-out test set the model never saw during
  training or tuning.
- Model + metadata (feature names, metrics, training date) are saved with
  joblib so the API layer can load them without needing this script.

Run:
    python model/train.py
Produces:
    model/churn_model.joblib
    model/metrics.json
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "model/churn_model.joblib"
METRICS_PATH = "model/metrics.json"
RANDOM_SEED = 42

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["churn"] = (df["churn"] == "Yes").astype(int)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",  # dataset is imbalanced (~33% churn)
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def find_best_threshold(y_true, y_proba) -> float:
    """Pick the probability threshold that maximizes F1, instead of assuming 0.5.
    For churn, missing a churner (false negative) is usually costlier than a false
    positive (an unnecessary retention offer), so this can be tuned further with
    a business-supplied cost ratio -- default here is F1-optimal as a sane baseline.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx])


def main():
    df = load_data()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_pipeline()

    # Cross-validation on training data only (test set stays untouched)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full training set
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    best_threshold = find_best_threshold(y_train, pipeline.predict_proba(X_train)[:, 1])
    y_pred_test = (y_proba_test >= best_threshold).astype(int)

    test_auc = roc_auc_score(y_test, y_proba_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Chosen threshold (F1-optimal on train): {best_threshold:.3f}")
    print(classification_report(y_test, y_pred_test))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # Feature importance (top drivers of churn -- useful for the README / stakeholder story)
    ohe_feature_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    all_feature_names = NUMERIC_FEATURES + list(ohe_feature_names)
    importances = pipeline.named_steps["model"].feature_importances_
    top_features = sorted(zip(all_feature_names, importances), key=lambda x: -x[1])[:10]

    # Save model artifact + metadata together, so the API never has to guess
    # what feature order/threshold/version the model expects.
    artifact = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test_roc_auc": float(test_auc),
        "threshold": best_threshold,
        "classification_report": report,
        "confusion_matrix": cm,
        "top_features": [{"feature": f, "importance": float(i)} for f, i in top_features],
        "trained_at": artifact["trained_at"],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
