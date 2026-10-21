"""Export the best completed tuning run for GitHub and Docker packaging."""

import argparse
import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
)
TUNING_METRICS = (
    "cv_recall_mean",
    "cv_recall_std",
    "cv_f1_mean",
    "cv_f1_std",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the top completed MLflow tuning run."
    )
    parser.add_argument(
        "--experiment",
        default="cardiovascular-disease-tuning-b2",
        help="MLflow experiment containing manual tuning runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ci_artifacts",
        help="Directory to download the selected run's model and reports.",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing_env = [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]
    if missing_env:
        raise RuntimeError(f"Variabel tracking/B2 belum diisi: {missing_env}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"Experiment tidak ditemukan: {args.experiment}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    eligible_runs = [
        run
        for run in runs
        if all(metric in run.data.metrics for metric in TUNING_METRICS)
    ]
    workflow_run_id = os.getenv("GITHUB_RUN_ID")
    if workflow_run_id:
        eligible_runs = [
            run
            for run in eligible_runs
            if run.data.params.get("github_run_id") == workflow_run_id
        ]
    if not eligible_runs:
        scope = f" untuk workflow run {workflow_run_id}" if workflow_run_id else ""
        raise RuntimeError(
            "Tidak ada run tuning FINISHED dengan metrik CV lengkap" + scope + "."
        )

    # Prioritize CV recall, then CV F1, then lower fold-to-fold variation.
    best_run = max(
        eligible_runs,
        key=lambda run: (
            run.data.metrics["cv_recall_mean"],
            run.data.metrics["cv_f1_mean"],
            -run.data.metrics["cv_recall_std"],
            -run.data.metrics["cv_f1_std"],
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(best_run.info.run_id, "model", str(args.output_dir))
    client.download_artifacts(
        best_run.info.run_id,
        "evaluation",
        str(args.output_dir),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    (args.output_dir / "model_uri.txt").write_text(
        model_uri + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment": args.experiment,
        "run_id": best_run.info.run_id,
        "model_uri": model_uri,
        "artifact_uri": best_run.info.artifact_uri,
        "params": best_run.data.params,
        "metrics": best_run.data.metrics,
    }
    (args.output_dir / "best_run.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Selected run: {best_run.info.run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Exported artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
