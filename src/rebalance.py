"""
rebalance.py — TP2 Part 3.9 : rebalance the TRAIN set only
==========================================================
Validation and test keep the real distribution (otherwise scores would be fake).
The original train.csv is kept unchanged; the result goes to train_balanced.csv.

Strategy (imbalanced-learn, the reference library):
  1. Random UNDERsampling of the majority class (non_toxic) down to 150
  2. Random OVERsampling of minority classes up to 100,
     but never more than x5 copies (threat: 6 -> 30), to limit overfitting
     on a few duplicated examples
In TP3 this can be compared with the other option: class weights in the loss.
"""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

MAJORITY_MAX = 150
MINORITY_TARGET = 100
MAX_FACTOR = 5
SEED = 42


def main():
    """Under-sample the majority, over-sample minorities, save train_balanced.csv, log to MLflow."""
    import mlflow

    train = pd.read_csv(PROC / "train.csv")
    before = train["label"].value_counts()

    # 1. undersample the majority class
    under = {c: min(n, MAJORITY_MAX) for c, n in before.items()}
    x, y = RandomUnderSampler(sampling_strategy=under, random_state=SEED).fit_resample(train, train["label"])

    # 2. oversample minorities (capped at MAX_FACTOR copies)
    mid = y.value_counts()
    over = {c: max(n, min(MINORITY_TARGET, n * MAX_FACTOR)) for c, n in mid.items()}
    x, y = RandomOverSampler(sampling_strategy=over, random_state=SEED).fit_resample(x, y)

    balanced = x.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle
    balanced.to_csv(PROC / "train_balanced.csv", index=False, encoding="utf-8")

    after = balanced["label"].value_counts()
    comp = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
    comp.loc["TOTAL"] = comp.sum()
    print(comp.to_string())
    print(f"\nimbalance ratio: {before.max() / before.min():.0f} -> {after.max() / after.min():.0f}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP2_Preparation")
    with mlflow.start_run(run_name="rebalance"):
        mlflow.log_params({"strategy": "RandomUnderSampler + RandomOverSampler (train only)",
                           "majority_max": MAJORITY_MAX, "minority_target": MINORITY_TARGET,
                           "max_factor": MAX_FACTOR, "seed": SEED})
        mlflow.log_metrics({"n_before": len(train), "n_after": len(balanced),
                            "ratio_before": before.max() / before.min(),
                            "ratio_after": after.max() / after.min()})
        mlflow.log_text(comp.to_string(), "rebalancing.txt")


if __name__ == "__main__":
    main()
