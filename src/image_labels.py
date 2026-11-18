"""
image_labels.py — TP2 Part 1.3 : image labels
=============================================
Hateful Memes already has labels made by trained annotators (Meta) for the
whole meme (picture + embedded text). Annotating memes by hand is slow and less
reliable, so we reuse them:   1 -> hateful_discriminatory,   0 -> neutral
Output: data/annotated/image_annotations.csv   (id | image_path | image_label | annotator_id)
"""

import json

import pandas as pd

from utils import DATA

MAPPING = {1: "hateful_discriminatory", 0: "neutral"}


def main():
    """Turn the dataset's expert labels into our image annotation file."""
    memes = pd.read_csv(DATA / "hateful_memes_raw.csv")
    labels = memes["metadata"].map(lambda m: MAPPING[json.loads(m)["original_label"]])
    out = pd.DataFrame({"id": memes["id"], "image_path": memes["media"],
                        "image_label": labels, "annotator_id": "meta_expert"})
    out.to_csv(DATA / "annotated" / "image_annotations.csv", index=False, encoding="utf-8")
    print(out["image_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
