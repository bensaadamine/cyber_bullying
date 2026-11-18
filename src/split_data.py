"""
split_data.py — TP2 Part 4 : train / validation / test (≈ 70 / 15 / 15)
=======================================================================
- Stratified  : class proportions preserved in every split
- Grouped     : posts of the same Reddit thread, or memes with the same caption,
                stay in the SAME split  -> no data leakage
Tool: scikit-learn StratifiedGroupKFold (stratification + groups in one object).
  test       = 1 fold out of 7   (≈ 14 %)
  validation = 1 fold out of 6 of the rest (≈ 14 %)
  train      = the rest          (≈ 72 %)

Input : data/processed/cyberbullying_processed.csv
Output: data/processed/train.csv, validation.csv, test.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 42


def take_one_fold(df, n_splits):
    """Return (rest, fold) using the first fold of a StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest_idx, fold_idx = next(sgkf.split(df, df["label"], groups=df["group"]))
    return df.iloc[rest_idx], df.iloc[fold_idx]


def main():
    """Split into train/validation/test, check for leakage, save the 3 CSVs, log to MLflow."""
    import mlflow

    df = pd.read_csv(PROC / "cyberbullying_processed.csv")
    rest, test = take_one_fold(df, n_splits=7)
    train, val = take_one_fold(rest, n_splits=6)
    splits = {"train": train, "validation": val, "test": test}

    # ---- leakage checks -----------------------------------------------------
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        g = set(splits[a]["group"]) & set(splits[b]["group"])
        t = set(splits[a]["text"]) & set(splits[b]["text"])
        assert not g, f"group leakage {a}/{b}: {list(g)[:3]}"
        assert not t, f"identical text in {a} and {b}"
    print("no group / text overlap between splits ✓\n")

    # ---- distribution -------------------------------------------------------
    dist = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()}).fillna(0).astype(int)
    dist.loc["TOTAL"] = dist.sum()
    print(dist.to_string())
    share = {k: f"{len(v) / len(df):.0%}" for k, v in splits.items()}
    print("\nsizes:", share)

    for name, part in splits.items():
        part.to_csv(PROC / f"{name}.csv", index=False, encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP2_Preparation")
    with mlflow.start_run(run_name="split"):
        mlflow.log_params({"method": "StratifiedGroupKFold (7 then 6 folds)",
                           "group": "reddit thread / meme caption", "seed": SEED})
        mlflow.log_metrics({f"n_{k}": len(v) for k, v in splits.items()})
        mlflow.log_text(dist.to_string(), "split_distribution.txt")


if __name__ == "__main__":
    main()
