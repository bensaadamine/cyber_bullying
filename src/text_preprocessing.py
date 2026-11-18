"""
text_preprocessing.py — TP2 Part 2 : text preprocessing
=======================================================
Starts again from the RAW text (TP1 cleaning removed too much for a model) and
KEEPS what helps detect irony and insults: emojis (as text), punctuation, hashtags.
Input : data/annotated/cyberbullying_annotated.csv, data/annotated/image_annotations.csv,
        data/cyberbullying_raw.csv
Output: data/processed/cyberbullying_processed.csv, reports/preprocessing_examples.md
"""

import html
import re
import unicodedata

import emoji
import pandas as pd

from utils import DATA, PROC, REPORTS, log_to_mlflow

MIN_WORDS = 2


def preprocess(text):
    """Normalise one raw text while keeping emojis, punctuation and hashtags."""
    t = unicodedata.normalize("NFKC", str(text))                 # unify unicode
    t = html.unescape(t)                                         # &amp; -> &
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)              # control characters
    t = re.sub(r"\[(removed|deleted)\]", " ", t, flags=re.I)     # reddit markers
    t = re.sub(r"!\[gif\]\([^)]*\)", " <gif> ", t)               # embedded gifs
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)               # [text](link) -> text
    t = re.sub(r"https?://\S+|www\.\S+", " <url> ", t)           # URL -> token
    t = re.sub(r"(?<!\w)(/?u/\w+|@\w+)", " <user> ", t)          # mention -> token
    t = re.sub(r"\^\((.*?)\)", r"\1", t)                         # ^(small text)
    t = re.sub(r"[*_~`]", "", t)                                 # markdown emphasis
    t = emoji.demojize(t, delimiters=(" :", ": "))               # 😂 -> :face_with_tears_of_joy:
    t = re.sub(r"[​-‏⁠﻿]", "", t)            # invisible characters
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)                       # "noooooo" -> "nooo"
    t = t.lower()                                                # our models are uncased
    return re.sub(r"\s+", " ", t).strip()


def group_of(row):
    """Posts of the same Reddit thread, or memes with the same caption, share a group."""
    if row["source"] == "reddit":
        return "rd_post_" + row["url"].split("/comments/")[1].split("/")[0]
    return "hm_text_" + re.sub(r"[^a-z0-9]", "", row["text_raw"].lower())[:60]


def main():
    """Join labels and raw text, preprocess, remove duplicates, save the processed CSV."""
    ann = pd.read_csv(DATA / "annotated" / "cyberbullying_annotated.csv")
    raw = pd.read_csv(DATA / "cyberbullying_raw.csv")[["id", "text", "url"]].rename(columns={"text": "text_raw"})
    img = pd.read_csv(DATA / "annotated" / "image_annotations.csv")[["id", "image_path", "image_label"]]
    df = ann.drop(columns="clean_text").merge(raw, on="id").merge(img, on="id", how="left")
    df[["image_path", "image_label"]] = df[["image_path", "image_label"]].fillna("")

    df["text"] = df["text_raw"].map(preprocess)
    n_in = len(df)

    # empty texts (rows with an image are kept), exact duplicates, duplicates after normalisation
    df = df[(df["text"].str.split().str.len() >= MIN_WORDS) | (df["image_path"] != "")]
    df = df.drop_duplicates(["text_raw", "image_path"])
    df["key"] = df["text"].str.replace(r"[^a-z0-9]", "", regex=True)
    df = df.drop_duplicates(["key", "image_path"])
    print(f"{n_in} annotated posts -> {len(df)} after removing empty texts and duplicates")

    df["group"] = df.apply(group_of, axis=1)
    cols = ["id", "text", "label", "ambiguous", "annotator_id", "source",
            "query_group", "image_path", "image_label", "group"]
    out = PROC / "cyberbullying_processed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out, index=False, encoding="utf-8")
    print(f"Saved -> {out}")

    # before / after examples: the 6 texts the pipeline changed the most
    df["change"] = (df["text_raw"].str.len() - df["text"].str.len()).abs()
    lines = ["# Prétraitement texte — exemples avant / après\n"]
    for _, r in df.nlargest(6, "change").iterrows():
        lines += [f"**{r['id']}**", f"- avant : `{r['text_raw'][:200]!r}`", f"- après : `{r['text'][:200]!r}`", ""]
    report = REPORTS / "preprocessing_examples.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    log_to_mlflow("TP2_Preparation", "text_preprocessing",
                  params={"min_words": MIN_WORDS, "urls": "<url>", "mentions": "<user>",
                          "emojis": "kept as text", "punctuation": "kept", "lowercase": True},
                  metrics={"n_in": n_in, "n_out": len(df)},
                  files=[out, report])


if __name__ == "__main__":
    main()
