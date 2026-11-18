"""
cleaning.py — TP1 Part 3 : merge the two sources and clean the text
===================================================================
Input : data/reddit_raw.csv + data/hateful_memes_raw.csv
Output: data/cyberbullying_raw.csv        (both sources, all columns)
        data/cyberbullying_raw_clean.csv  (id, date, clean_text, media, url, source)
"""

import html
import re
import unicodedata

import emoji
import pandas as pd

from utils import DATA, log_to_mlflow

MIN_WORDS = 3            # shorter texts carry too little information
SEXUAL_CAP = 0.20        # max share of sexual texts in the corpus
SEED = 42

BOT_MESSAGES = re.compile(r"i am a bot|thank you for submitting|your submission has been removed|"
                          r"this action was performed automatically|please contact the moderators", re.I)
# used to FLAG sexual content (not to delete it). "rape" is left out on purpose: it signals threats.
SEXUAL_WORDS = re.compile(r"\b(?:sex|sexy|sexual|cock|dick|pussy|tits|boobs?|cum|horny|porn|nudes?|"
                          r"slut|whore|blowjob|virgin|femboy)s?\b", re.I)


def clean_text(text):
    """Clean one text: unicode, URLs, mentions, markdown, emojis to text, lowercase."""
    t = unicodedata.normalize("NFKC", str(text))
    t = html.unescape(t)                                      # &amp; -> &
    t = re.sub(r"\[(removed|deleted)\]", " ", t, flags=re.I)  # reddit markers
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)              # URLs
    t = re.sub(r"(?<!\w)(/?u/\w+|@\w+)", " ", t)              # mentions (personal data)
    t = re.sub(r"#(\w+)", r"\1", t)                           # #word -> word
    t = re.sub(r"\^\((.*?)\)", r"\1", t)                      # ^(small text)
    t = re.sub(r"[*_~`]|^>+", " ", t, flags=re.M)             # markdown symbols
    t = emoji.demojize(t, delimiters=(" :", ": "))            # 😂 -> :face_with_tears_of_joy:
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def main():
    """Merge the sources, apply the cleaning steps, save both CSVs, log to MLflow."""
    raw = pd.concat([pd.read_csv(DATA / "reddit_raw.csv"),
                     pd.read_csv(DATA / "hateful_memes_raw.csv")], ignore_index=True)
    raw["media"] = raw["media"].fillna("")
    raw.to_csv(DATA / "cyberbullying_raw.csv", index=False, encoding="utf-8")
    print("raw rows:", len(raw))

    df = raw.copy()
    removed = {}

    # 1. bot / moderator messages
    bots = df["text"].fillna("").str.contains(BOT_MESSAGES)
    removed["bots"] = int(bots.sum())
    df = df[~bots]

    # 2. clean the text, then drop texts that are too short (rows with an image are kept)
    df["clean_text"] = df["text"].fillna("").map(clean_text)
    short = (df["clean_text"].str.split().str.len() < MIN_WORDS) & (df["media"] == "")
    removed["short"] = int(short.sum())
    df = df[~short]

    # 3. duplicates (same text AND same image)
    dup = df.duplicated(["clean_text", "media"])
    removed["duplicates"] = int(dup.sum())
    df = df[~dup]

    # 4. sexual content: if above 20 %, drop random sexual text-only rows
    df["is_sexual"] = df["clean_text"].str.contains(SEXUAL_WORDS)
    allowed = int(SEXUAL_CAP * (~df["is_sexual"]).sum() / (1 - SEXUAL_CAP))
    extra = max(0, int(df["is_sexual"].sum()) - allowed)
    candidates = df[df["is_sexual"] & (df["media"] == "")]
    drop = candidates.sample(min(extra, len(candidates)), random_state=SEED).index
    removed["sexual_cap"] = len(drop)
    df = df.drop(drop)

    print("removed:", removed, "| sexual share:", f"{df['is_sexual'].mean():.1%}")
    out = DATA / "cyberbullying_raw_clean.csv"
    df[["id", "date", "clean_text", "media", "url", "source"]].to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {out}")

    log_to_mlflow("TP1_Collecte", "cleaning",
                  params={"min_words": MIN_WORDS, "sexual_cap": SEXUAL_CAP, "seed": SEED},
                  metrics={"rows_raw": len(raw), "rows_clean": len(df),
                           "sexual_share": float(df["is_sexual"].mean()),
                           **{f"removed_{k}": v for k, v in removed.items()}},
                  files=[DATA / "cyberbullying_raw.csv", out])


if __name__ == "__main__":
    main()
