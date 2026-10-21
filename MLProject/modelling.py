"""Tune MLflow models using the preprocessed train/test CSV files in this folder."""

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "cardiovascular-disease_preprocessing"
ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5


def load_processed_data(data_dir: Path):
    """Load and validate the preprocessed train/test matrices and targets."""
    files = {
        "X_train": data_dir / "X_train.csv",
        "X_test": data_dir / "X_test.csv",
        "y_train": data_dir / "y_train.csv",
        "y_test": data_dir / "y_test.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Dataset preprocessing tidak lengkap; letakkan artifact B2 di "
            f"{data_dir}:\n" + "\n".join(missing)
        )

    X_train = pd.read_csv(files["X_train"])
    X_test = pd.read_csv(files["X_test"])
    y_train_frame = pd.read_csv(files["y_train"])
    y_test_frame = pd.read_csv(files["y_test"])

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Kolom X_train dan X_test tidak sama.")
    if list(y_train_frame.columns) != ["cardio"]:
        raise ValueError("y_train.csv harus memiliki kolom cardio.")
    if list(y_test_frame.columns) != ["cardio"]:
        raise ValueError("y_test.csv harus memiliki kolom cardio.")

    y_train = y_train_frame["cardio"]
    y_test = y_test_frame["cardio"]
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Jumlah baris feature dan target tidak cocok.")
    return X_train, X_test, y_train, y_test


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    positive_index = classes.index(1)
    y_score = probabilities[:, positive_index]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        "test_pr_auc": float(average_precision_score(y_test, y_score)),
        "test_log_loss": float(log_loss(y_test, y_score, labels=[0, 1])),
        "test_false_positives": float(fp),
        "test_false_negatives": float(fn),
        "test_true_positives": float(tp),
        "test_true_negatives": float(tn),
    }
    return metrics, y_pred, y_score


def log_evaluation_artifacts(y_test, y_pred, y_score, cv_results, metrics, model_name):
    """Save manual evaluation artifacts beyond the metrics logged to MLflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        confusion_path = temp_path / "confusion_matrix.png"
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=[0, 1],
            cmap="Blues",
            values_format="d",
        )
        display.ax_.set_title(f"Confusion Matrix - {model_name}")
        display.figure_.tight_layout()
        display.figure_.savefig(confusion_path, dpi=150, bbox_inches="tight")
        plt.close(display.figure_)
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")

        curves_path = temp_path / "roc_pr_curves.png"
        fpr, tpr, _ = roc_curve(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
        axes[1].plot(recall, precision)
        axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
        figure.suptitle(f"Evaluation Curves - {model_name}")
        figure.tight_layout()
        figure.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        mlflow.log_artifact(str(curves_path), artifact_path="evaluation")

        cv_path = temp_path / "cv_results.csv"
        cv_results.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), artifact_path="evaluation")
        mlflow.log_dict(
            {
                "positive_class": 1,
                "decision_threshold": DECISION_THRESHOLD,
                "selection_metric": "mean 5-fold CV recall",
                "metrics": metrics,
                "definitions": {
                    "recall": "TP / (TP + FN); primary metric to reduce false negatives.",
                    "f1": "Harmonic mean of precision and recall.",
                    "roc_auc": "Area under the ROC Curve.",
                    "pr_auc": "Average precision over the precision-recall curve.",
                },
            },
            "evaluation/metric_info.json",
        )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing_env = [
        name
        for name in ENV_VARS
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]
    if missing_env:
        raise RuntimeError(f"Variable MLflow belum diisi: {missing_env}")

    data_dir = Path(os.getenv("CARDIO_PREPROCESSED_DATA_DIR", str(DATA_DIR)))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "f1": "f1"}
    model_searches = {
        "logistic-regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            [
                {
                    "n_estimators": [50],
                    "max_depth": [8],
                    "min_samples_leaf": [2],
                    "class_weight": [None],
                },
                {
                    "n_estimators": [100],
                    "max_depth": [12],
                    "min_samples_leaf": [2],
                    "class_weight": ["balanced_subsample"],
                },
            ],
        ),
    }

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("cardiovascular-disease-tuning-b2")

    for model_name, (estimator, param_grid) in model_searches.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit="recall",
            cv=cv,
            n_jobs=1,
            verbose=1,
            return_train_score=False,
            error_score="raise",
        )
        with mlflow.start_run(run_name=f"submission-tuning-{model_name}") as run:
            start_time = time.perf_counter()
            search.fit(X_train, y_train)
            search_seconds = time.perf_counter() - start_time
            best_index = search.best_index_
            cv_results = pd.DataFrame(search.cv_results_)
            best_model = search.best_estimator_
            test_metrics, y_pred, y_score = evaluate_on_test(
                best_model, X_test, y_test
            )
            metrics = {
                **test_metrics,
                "cv_recall_mean": float(
                    search.cv_results_["mean_test_recall"][best_index]
                ),
                "cv_recall_std": float(
                    search.cv_results_["std_test_recall"][best_index]
                ),
                "cv_f1_mean": float(search.cv_results_["mean_test_f1"][best_index]),
                "cv_f1_std": float(search.cv_results_["std_test_f1"][best_index]),
                "search_fit_time_seconds": float(search_seconds),
            }
            params = {
                "model_name": model_name,
                "cv_type": "StratifiedKFold",
                "cv_splits": 5,
                "cv_selection_metric": "recall",
                "decision_threshold": DECISION_THRESHOLD,
                **{f"best_{key}": str(value) for key, value in search.best_params_.items()},
            }
            if os.getenv("GITHUB_RUN_ID"):
                params["github_run_id"] = os.environ["GITHUB_RUN_ID"]

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            input_example = X_train.head(1)
            signature = mlflow.models.infer_signature(
                input_example,
                best_model.predict(input_example),
            )
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )
            log_evaluation_artifacts(
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
                cv_results=cv_results,
                metrics=metrics,
                model_name=model_name,
            )

            print(f"Model: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Best params: {search.best_params_}")
            print(f"CV recall: {metrics['cv_recall_mean']:.4f}")
            print(f"Test F1: {metrics['test_f1']:.4f}")


if __name__ == "__main__":
    # MLflow Projects starts an orchestration run; tuning creates its own runs.
    os.environ.pop("MLFLOW_RUN_ID", None)
    main()
