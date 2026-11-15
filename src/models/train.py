import argparse
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features.preprocess import FEATURES, build_preprocessor

CFG = yaml.safe_load(open("configs/params.yaml"))
TARGET = CFG["split"]["target"]
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "telco-churn"
MODEL_NAME = "telco-churn"
BASE_TRUSTED_TYPES = ["numpy.dtype", "src.features.preprocess.clean"]
EXTRA_TRUSTED_TYPES = {"random_forest": ["sklearn.tree._tree.Tree"]}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_model(model_type: str, seed: int):
    p = CFG["train"][model_type]
    if model_type == "logistic_regression":
        return LogisticRegression(class_weight="balanced", random_state=seed, **p)
    if model_type == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **p)
    raise ValueError(f"Bilinmeyen model: {model_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", default=["batch_1"])
    ap.add_argument("--model-type", default=CFG["train"]["model_type"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--register", action="store_true", help="Modeli registry'ye kaydet")
    args = ap.parse_args()

    seed = CFG["train"]["seed"]
    train_df = pd.concat([pd.read_csv(f"data/batches/{b}.csv") for b in args.batches])
    test_df = pd.read_csv("data/batches/test_holdout.csv")

    X_train, y_train = train_df.drop(columns=[TARGET]), (train_df[TARGET] == "Yes").astype(int)
    X_test, y_test = test_df.drop(columns=[TARGET]), (test_df[TARGET] == "Yes").astype(int)

    model = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", make_model(args.model_type, seed)),
    ])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model_type": args.model_type,
            "data_batches": "+".join(args.batches),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "seed": seed,
            **CFG["train"][args.model_type],
        })
        mlflow.set_tags({"git_commit": git_commit()})

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "accuracy": accuracy_score(y_test, pred),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, pred)
        trusted = BASE_TRUSTED_TYPES + EXTRA_TRUSTED_TYPES.get(args.model_type, [])
        mlflow.sklearn.log_model(
    model,
    name="model",
    signature=signature,
    skops_trusted_types=trusted,
    registered_model_name=MODEL_NAME if args.register else None,
)
        print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()