"""
split_data.py — TP2 Part 4 : train / validation / test (about 70 / 15 / 15)
==========================================================================
StratifiedGroupKFold (scikit-learn) does both jobs at once:
- stratified: every split keeps the class proportions
- grouped: posts of the same group (Reddit thread / meme caption) stay together -> no leakage
test = 1 fold out of 7 (~14 %), validation = 1 fold out of 6 of the rest (~14 %), train = the rest.
Output: data/processed/train.csv, validation.csv, test.csv
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils import PROC, log_to_mlflow

SEED = 42


def take_one_fold(df, n_splits):
    """Split df into (rest, one fold) with a stratified, grouped fold."""
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest_idx, fold_idx = next(folds.split(df, df["label"], groups=df["group"]))
    return df.iloc[rest_idx], df.iloc[fold_idx]


def main():
    """Make the 3 splits, check there is no leakage, save them."""
    df = pd.read_csv(PROC / "cyberbullying_processed.csv")
    rest, test = take_one_fold(df, 7)
    train, val = take_one_fold(rest, 6)
    splits = {"train": train, "validation": val, "test": test}

    # leakage check: no group shared by two splits
    assert not set(train["group"]) & set(val["group"]), "leakage train/validation"
    assert not set(train["group"]) & set(test["group"]), "leakage train/test"
    assert not set(val["group"]) & set(test["group"]), "leakage validation/test"

    for name, part in splits.items():
        part.to_csv(PROC / f"{name}.csv", index=False, encoding="utf-8")
    table = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()}).fillna(0).astype(int)
    print(table.to_string())

    log_to_mlflow("TP2_Preparation", "split",
                  params={"method": "StratifiedGroupKFold", "seed": SEED},
                  metrics={f"n_{k}": len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
