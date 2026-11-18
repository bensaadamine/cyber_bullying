"""
rebalance.py — TP2 Part 3.9 : rebalance the TRAIN set only
==========================================================
Validation and test keep the real distribution. train.csv is kept unchanged.
1. undersample the majority class (non_toxic) down to 150
2. oversample the other classes up to 100, but never more than 5 copies
   of the same post (threat: 6 -> 30), to limit overfitting
Output: data/processed/train_balanced.csv
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from utils import PROC, log_to_mlflow

MAJORITY_MAX, MINORITY_TARGET, MAX_FACTOR, SEED = 150, 100, 5, 42


def main():
    """Under-sample the majority, over-sample minorities, save train_balanced.csv."""
    train = pd.read_csv(PROC / "train.csv")
    before = train["label"].value_counts()

    under = {c: min(n, MAJORITY_MAX) for c, n in before.items()}
    x, y = RandomUnderSampler(sampling_strategy=under, random_state=SEED).fit_resample(train, train["label"])
    over = {c: max(n, min(MINORITY_TARGET, n * MAX_FACTOR)) for c, n in y.value_counts().items()}
    x, y = RandomOverSampler(sampling_strategy=over, random_state=SEED).fit_resample(x, y)

    balanced = x.sample(frac=1, random_state=SEED)               # shuffle the rows
    balanced.to_csv(PROC / "train_balanced.csv", index=False, encoding="utf-8")
    after = balanced["label"].value_counts()
    print(pd.DataFrame({"before": before, "after": after}).to_string())

    log_to_mlflow("TP2_Preparation", "rebalance",
                  params={"majority_max": MAJORITY_MAX, "minority_target": MINORITY_TARGET,
                          "max_factor": MAX_FACTOR, "seed": SEED},
                  metrics={"n_before": len(train), "n_after": len(balanced),
                           "ratio_before": before.max() / before.min(),
                           "ratio_after": after.max() / after.min()})


if __name__ == "__main__":
    main()
