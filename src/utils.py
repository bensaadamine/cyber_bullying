"""utils.py — paths and the MLflow helper shared by every script."""

from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]      # project folder (TP_FR)
DATA = ROOT / "data"
PROC = DATA / "processed"
REPORTS = ROOT / "reports"


def log_to_mlflow(experiment, run_name, params=None, metrics=None, files=()):
    """Save one run in MLflow: its parameters, its metrics and some files."""
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics or {})
        for f in files:
            mlflow.log_artifact(str(f))
