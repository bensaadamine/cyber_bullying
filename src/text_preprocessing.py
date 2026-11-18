"""
text_preprocessing.py — TP2 Part 2 (text) : build the processed dataset
=======================================================================
Starts again from the RAW text (TP1 cleaning was too aggressive for models:
it removed hashtags, mentions...). Here we KEEP what helps detect irony and
offence: emojis (as text), punctuation, "!!!", hashtags.

Input : data/annotated/cyberbullying_annotated.csv   (labels)
        data/annotated/image_annotations.csv         (image labels)
        data/cyberbullying_raw.csv                   (raw text, url)
Output: data/processed/cyberbullying_processed.csv
        reports/preprocessing_examples.md            (before / after)
"""

import html
import re
import unicodedata
from pathlib import Path

import emoji
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed" / "cyberbullying_processed.csv"
REPORT = ROOT / "reports" / "preprocessing_examples.md"
MIN_WORDS = 2


# --------------------------------------------------------------------------
# One function = the whole text pipeline, one documented line per operation
# --------------------------------------------------------------------------
def preprocess(text):
    """Normalise one raw text while keeping emojis, punctuation and hashtags."""
    t = unicodedata.normalize("NFKC", str(text))                 # unify unicode
    t = html.unescape(t)                                         # &amp; -> &
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)              # control chars
    t = re.sub(r"\[(removed|deleted)\]", " ", t, flags=re.I)     # reddit markers
    t = re.sub(r"!\[gif\]\([^)]*\)", " <gif> ", t)               # embedded gifs
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)                # [text](link) -> text
    t = re.sub(r"https?://\S+|www\.\S+", " <url> ", t)           # URL -> token
    t = re.sub(r"(?<!\w)(/?u/\w+|@\w+)", " <user> ", t)          # mention -> token (PII)
    t = re.sub(r"\^\((.*?)\)", r"\1", t)                         # ^(small text)
    t = re.sub(r"[*_~`]", "", t)                                 # markdown emphasis
    t = emoji.demojize(t, delimiters=(" :", ": "))               # 😂 -> :face_with_tears_of_joy:
    t = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", t)            # zero-width chars (after emojis)
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)                       # "noooooo" -> "nooo", "!!!!!" -> "!!!"
    t = t.lower()                                                # models are uncased
    t = re.sub(r"\s+", " ", t).strip()                           # whitespace
    return t


def norm_key(t):
    """Key for 'duplicates after normalisation': letters and digits only."""
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())


def group_key(row):
    """Posts sharing a group must stay in the same split (TP2 Part 4, leakage)."""
    if row["source"] == "reddit":
        m = re.search(r"/comments/([a-z0-9]+)/", row["url"])
        return f"rd_post_{m.group(1)}" if m else row["id"]
    return f"hm_text_{norm_key(row['text_raw'])[:60]}"   # same caption = same group


def step(df, keep, name, log):
    """Apply a filter, print and record how many rows it removed."""
    log[name] = int((~keep).sum())
    print(f"{name:<32} -{log[name]:>4} -> {int(keep.sum())}")
    return df[keep].copy()


def main():
    """Join labels and raw text, preprocess, deduplicate, save the processed CSV, log to MLflow."""
    import mlflow

    ann = pd.read_csv(DATA / "annotated" / "cyberbullying_annotated.csv")
    raw = pd.read_csv(DATA / "cyberbullying_raw.csv")[["id", "text", "url"]]
    img = pd.read_csv(DATA / "annotated" / "image_annotations.csv")

    df = (ann.drop(columns=["clean_text"])
          .merge(raw.rename(columns={"text": "text_raw"}), on="id", how="left")
          .merge(img[["id", "image_path", "image_label"]], on="id", how="left"))
    df["image_path"] = df["image_path"].fillna("")
    df["image_label"] = df["image_label"].fillna("")
    print(f"{'annotated posts':<32} {len(df):>5}")

    log = {}
    df["text"] = df["text_raw"].map(preprocess)

    # 1. empty / unusable (posts with an image are kept: the image carries info)
    n_words = df["text"].str.split().str.len().fillna(0)
    df = step(df, (n_words >= MIN_WORDS) | (df["image_path"] != ""), "empty / unusable", log)

    # 2. exact duplicates (same raw text and same image)
    df = step(df, ~df.duplicated(["text_raw", "image_path"]), "exact duplicates", log)

    # 3. duplicates after normalisation (case, punctuation, spaces)
    df["_key"] = df["text"].map(norm_key)
    df = step(df, ~df.duplicated(["_key", "image_path"]), "normalised duplicates", log)

    df["group"] = df.apply(group_key, axis=1)

    # linguistic elements kept (proof we did not strip them)
    kept = {
        "with_emoji": df["text"].str.contains(r":[a-z_]+:").mean(),
        "with_exclamation": df["text"].str.contains("!").mean(),
        "with_question": df["text"].str.contains(r"\?").mean(),
        "with_hashtag": df["text"].str.contains(r"#\w").mean(),
    }
    print("kept elements:", {k: f"{v:.1%}" for k, v in kept.items()})

    cols = ["id", "text", "label", "ambiguous", "annotator_id", "source",
            "query_group", "image_path", "image_label", "group"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} posts -> {OUT}")

    # before / after examples: the posts the pipeline changed the most
    changed = df[df["text_raw"].str.lower().str.strip() != df["text"]].copy()
    changed["delta"] = (changed["text_raw"].str.len() - changed["text"].str.len()).abs()
    ex = changed.sort_values("delta", ascending=False).head(6)
    lines = ["# Prétraitement texte — exemples avant / après\n"]
    for _, r in ex.iterrows():
        lines += [f"**{r['id']}**", f"- avant : `{r['text_raw'][:200]!r}`",
                  f"- après : `{r['text'][:200]!r}`", ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP2_Preparation")
    with mlflow.start_run(run_name="text_preprocessing"):
        mlflow.log_params({"min_words": MIN_WORDS, "lowercase": True,
                           "emojis": "demojized", "urls": "<url>", "mentions": "<user>",
                           "elongation": "max 3 repeats"})
        mlflow.log_metrics({"n_in": len(ann), "n_out": len(df),
                            **{f"removed_{k.split()[0]}": v for k, v in log.items()},
                            **{f"share_{k}": float(v) for k, v in kept.items()}})
        mlflow.log_artifact(str(OUT), "data")
        mlflow.log_artifact(str(REPORT))


if __name__ == "__main__":
    main()
