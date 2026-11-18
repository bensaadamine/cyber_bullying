"""
select_for_annotation.py — TP2 Part 1 : choose the 600 posts to annotate
=======================================================================
All 200 memes (needed for the image model) + 400 Reddit posts spread equally
over the query groups (so every class has a chance to appear).
Output: data/annotated/to_annotate.csv
"""

import json

import pandas as pd

from utils import DATA

N_REDDIT = 400
SEED = 42


def main():
    """Pick all memes + 400 Reddit posts spread over query groups, save the list."""
    clean = pd.read_csv(DATA / "cyberbullying_raw_clean.csv")
    meta = pd.read_csv(DATA / "cyberbullying_raw.csv").set_index("id")["metadata"]
    clean["query_group"] = clean["id"].map(meta.map(lambda m: json.loads(m).get("query_group")))

    memes = clean[clean["source"] == "hateful_memes"]
    reddit = clean[clean["source"] == "reddit"]
    per_group = N_REDDIT // reddit["query_group"].nunique()
    picked = reddit.groupby("query_group").sample(per_group, random_state=SEED)
    rest = reddit.drop(picked.index).sample(N_REDDIT - len(picked), random_state=SEED)   # fill up to 400

    sel = pd.concat([memes, picked, rest])
    out = DATA / "annotated" / "to_annotate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sel[["id", "clean_text", "media", "url", "source", "query_group"]].to_csv(out, index=False, encoding="utf-8")
    print(sel["query_group"].value_counts().to_string())
    print(f"\n{len(sel)} posts -> {out}")


if __name__ == "__main__":
    main()
