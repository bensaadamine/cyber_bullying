"""
cleaning.py — TP1 Part 3 : cleaning and structuring
===================================================
Input : data/reddit_raw.csv
Output: data/cyberbullying_raw.csv        (raw corpus, all metadata kept)
        data/cyberbullying_raw_clean.csv  (id, date, clean_text, media, url, source)

Every step prints how many rows it removed -> numbers for the mini-report,
and the same numbers are logged to MLflow.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

import emoji
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MIN_WORDS = 3           # shorter texts carry too little information
SEXUAL_CAP = 0.20       # max share of sexual-mockery texts (see README)
SEED = 42

# messages written by bots / moderators, not by users
BOT_PATTERNS = re.compile(
    r"i am a bot|thank you for submitting|your submission has been removed|"
    r"this action was performed automatically|please contact the moderators",
    re.I,
)

# small lexicon to FLAG sexual content (not to delete it).
# "rape" is deliberately absent: it usually signals a threat, which we want.
SEXUAL_WORDS = re.compile(
    r"\b(?:sex|sexy|sexual|cock|dick|pussy|tits|boobs?|cum|horny|porn|nudes?|"
    r"slut|whore|blowjob|virgin|femboy)s?\b",
    re.I,
)


# --------------------------------------------------------------------------
# Text cleaning: one function, each line = one documented operation
# --------------------------------------------------------------------------
def clean_text(text):
    t = unicodedata.normalize("NFKC", str(text))              # unify unicode
    t = html.unescape(t)                                      # &amp; -> &
    t = re.sub(r"\[(removed|deleted)\]", " ", t, flags=re.I)  # reddit markers
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)              # URLs
    t = re.sub(r"(?<!\w)(/?u/\w+|@\w+)", " ", t)              # mentions (= PII)
    t = re.sub(r"#(\w+)", r"\1", t)                           # #word -> word
    t = re.sub(r"\^\((.*?)\)", r"\1", t)                       # ^(small text)
    t = re.sub(r"[*_~`]|^>+", " ", t, flags=re.M)             # markdown symbols
    t = emoji.demojize(t, delimiters=(" :", ": "))            # 😂 -> :face_with_tears_of_joy:
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()                        # whitespace
    return t


def step(df, mask_keep, name, log):
    """Apply a filter and record how many rows it removed."""
    removed = int((~mask_keep).sum())
    log[name] = removed
    print(f"{name:<28} -{removed:>4}  -> {int(mask_keep.sum())} rows")
    return df[mask_keep].copy()


def main():
    import mlflow

    raw = pd.read_csv(DATA_DIR / "reddit_raw.csv")
    raw["media"] = raw["media"].fillna("")

    # Part 5 structure: the raw corpus is saved under its official name
    raw.to_csv(DATA_DIR / "cyberbullying_raw.csv", index=False, encoding="utf-8")
    print(f"{'raw rows':<28} {len(raw):>5}")

    log = {}
    df = raw.copy()

    # 1. bot / moderator messages
    df = step(df, ~df["text"].fillna("").str.contains(BOT_PATTERNS), "bot/mod messages", log)

    # 2. clean the text
    df["clean_text"] = df["text"].fillna("").map(clean_text)

    # 3. empty or too short (image posts are kept: the image is the content)
    n_words = df["clean_text"].str.split().str.len()
    df = step(df, (n_words >= MIN_WORDS) | (df["media"] != ""), "empty / too short", log)

    # 4. duplicates (same text AND same image)
    df = step(df, ~df.duplicated(subset=["clean_text", "media"]), "duplicates", log)

    # 5. sexual-mockery cap: drop random sexual TEXT rows above 20 %
    df["is_sexual"] = df["clean_text"].str.contains(SEXUAL_WORDS)
    share_before = df["is_sexual"].mean()
    max_sexual = int(SEXUAL_CAP * (~df["is_sexual"]).sum() / (1 - SEXUAL_CAP))
    candidates = df[df["is_sexual"] & (df["media"] == "")]
    n_extra = int(df["is_sexual"].sum()) - max_sexual
    to_drop = candidates.sample(n=max(0, min(n_extra, len(candidates))), random_state=SEED).index
    df = step(df, ~df.index.isin(to_drop), "sexual cap (20%)", log)
    share_after = df["is_sexual"].mean()
    print(f"sexual share: {share_before:.1%} -> {share_after:.1%}")

    # 6. keep only the relevant columns
    #    (id kept on purpose: TP2 annotation needs a unique key)
    clean = df[["id", "date", "clean_text", "media", "url", "source"]]
    out = DATA_DIR / "cyberbullying_raw_clean.csv"
    clean.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(clean)} rows -> {out}")

    # ---- MLflow ----------------------------------------------------------
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP1_Collecte")
    with mlflow.start_run(run_name="cleaning"):
        mlflow.log_params({"min_words": MIN_WORDS, "sexual_cap": SEXUAL_CAP,
                           "seed": SEED, "emojis": "demojized (kept as text)",
                           "hashtags": "symbol removed, word kept"})
        mlflow.log_metrics({"rows_raw": len(raw), "rows_clean": len(clean),
                            "sexual_share_before": share_before,
                            "sexual_share_after": share_after,
                            **{f"removed_{k.split()[0].replace('/', '_')}": v
                               for k, v in log.items()}})
        mlflow.log_artifact(str(DATA_DIR / "cyberbullying_raw.csv"), "data")
        mlflow.log_artifact(str(out), "data")


if __name__ == "__main__":
    main()
