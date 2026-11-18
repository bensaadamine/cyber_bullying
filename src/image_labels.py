"""
image_labels.py — TP2 Part 1.3 : image labels from the Hateful Memes experts
===========================================================================
Manual image annotation was slow and uncertain. Hateful Memes already ships
labels made by trained annotators (Meta, Kiela et al. 2020) for the meme as a
whole (picture + embedded text) -> exactly our "meme as seen" rule.

Mapping:  original_label 1 -> hateful_discriminatory
          original_label 0 -> neutral

Quality check: the images already annotated by hand (A_human_1) are compared
with the expert labels (accuracy + Cohen's kappa, binary: hateful vs not).

Output: data/annotated/image_annotations.csv          (final, 200 images)
        data/annotated/image_annotations_human.csv    (manual sample, kept)
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANN = DATA / "annotated"
FINAL = ANN / "image_annotations.csv"
HUMAN = ANN / "image_annotations_human.csv"

MAPPING = {1: "hateful_discriminatory", 0: "neutral"}


def main():
    # 1. keep the manual sample under its own name (only the first time)
    """Turn the dataset's expert labels into our image annotation file."""
    if FINAL.exists() and not HUMAN.exists():
        manual = pd.read_csv(FINAL)
        if (manual["annotator_id"] != "meta_expert").all():
            manual.to_csv(HUMAN, index=False, encoding="utf-8")

    # 2. expert labels -> final image annotation file
    memes = pd.read_csv(DATA / "hateful_memes_raw.csv")
    orig = memes["metadata"].map(lambda m: json.loads(m)["original_label"])
    final = pd.DataFrame({
        "id": memes["id"],
        "image_path": memes["media"],
        "image_label": orig.map(MAPPING),
        "annotator_id": "meta_expert",
    })
    final.to_csv(FINAL, index=False, encoding="utf-8")
    print(final["image_label"].value_counts().to_string())

    # 3. agreement between the manual sample and the experts
    if HUMAN.exists():
        h = pd.read_csv(HUMAN).merge(final, on="id", suffixes=("_human", "_expert"))
        human_bin = h["image_label_human"].eq("hateful_discriminatory")
        expert_bin = h["image_label_expert"].eq("hateful_discriminatory")
        acc = (human_bin == expert_bin).mean()
        kappa = cohen_kappa_score(human_bin, expert_bin)
        print(f"\nManual sample vs experts ({len(h)} images): "
              f"agreement {acc:.0%}, Cohen's kappa {kappa:.2f}")
        print(pd.crosstab(h["image_label_human"], h["image_label_expert"]).to_string())


if __name__ == "__main__":
    main()
