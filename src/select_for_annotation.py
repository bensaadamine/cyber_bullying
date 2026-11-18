"""
select_for_annotation.py — TP2 Part 1 : choose the 600 posts to annotate
=======================================================================
- all memes (Hateful Memes, text + image)        -> needed for TP3 multimodal
- 400 Reddit posts, equally spread over query groups -> every class can appear

Output: data/annotated/to_annotate.csv
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "annotated" / "to_annotate.csv"

N_REDDIT = 400
SEED = 42


def main():
    """Pick all memes + 400 Reddit posts spread over query groups, save the list."""
    clean = pd.read_csv(DATA / "cyberbullying_raw_clean.csv")
    meta = (pd.read_csv(DATA / "cyberbullying_raw.csv")
            .set_index("id")["metadata"].map(json.loads))
    clean["query_group"] = clean["id"].map(meta.map(lambda m: m.get("query_group")))
    clean["media"] = clean["media"].fillna("")

    memes = clean[clean["source"] == "hateful_memes"]
    reddit = clean[clean["source"] == "reddit"]

    groups = sorted(reddit["query_group"].unique())
    per_group = N_REDDIT // len(groups)
    picked = (reddit.groupby("query_group", group_keys=False)
              .sample(per_group, random_state=SEED))
    rest = reddit.drop(picked.index).sample(N_REDDIT - len(picked), random_state=SEED)

    sel = pd.concat([memes, picked, rest]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sel[["id", "clean_text", "media", "url", "source", "query_group"]].to_csv(
        OUT, index=False, encoding="utf-8")
    print(sel["query_group"].value_counts().to_string())
    print(f"\n{len(sel)} posts -> {OUT}")


if __name__ == "__main__":
    main()
