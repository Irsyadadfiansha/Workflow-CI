

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "heart_preprocessing",
    "heart_preprocessed.csv"
)
TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "heart-disease-basic"

DAGSHUB_OWNER = os.getenv("DAGSHUB_OWNER", "irsyadadfiansha9")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO",  "heart-disease-mlops")


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    logger.info(f"Data loaded. Shape: {df.shape}")
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Enable autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC-AUC:  {auc:.4f}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

    logger.info("Training complete. Run: mlflow ui  to view results.")


if __name__ == "__main__":
    main()