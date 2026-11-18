"""
reddit_scraper.py — TP1 : Reddit collection via the Arctic Shift archive API
=============================================================================
Why Arctic Shift?  Reddit stopped accepting new API keys (PRAW) on 2026-10-31.
Arctic Shift is a free, keyless, public archive of Reddit (Pushshift successor).
Docs: https://github.com/ArthurHeitmann/arctic_shift/tree/master/api

Output: data/reddit_raw.csv  with columns
    id | date | text | media | url | source | metadata
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# --------------------------------------------------------------------------
# 1. CONFIGURATION  (everything here is logged to MLflow -> reproducibility)
# --------------------------------------------------------------------------
BASE_URL = "https://arctic-shift.photon-reddit.com/api"
ROOT = Path(__file__).resolve().parents[1]          # project root
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

DATE_AFTER = "2026-05-01"    # collection window
DATE_BEFORE = "2026-11-01"   # (>36h old so scores are final)
PER_QUERY = 60               # max comments kept per (subreddit, group)
SALT = "tp-fr-2026"          # salt for anonymising usernames

# keyword groups -> postgres websearch syntax ("a b" = phrase, OR = or)
# NOTE: no sexual words on purpose (see README: sexual mockery is capped)
KEYWORDS = {
    "insult":  'ugly OR loser OR stupid OR idiot OR pathetic OR fat',
    "threat":  '"kill yourself" OR kys OR "watch your back" OR "find you" OR "hurt you"',
    "hate":    'subhuman OR "go back to" OR disgusting OR vermin OR worthless',
    "irony":   'cope OR seethe OR ratio OR "nobody asked" OR "ok boomer"',
    "neutral": None,          # no keyword: random recent comments
}

SUBREDDITS = {
    "insult":  ["RoastMe", "gaming", "4chan"],
    "threat":  ["bullying", "TrueOffMyChest", "4chan"],
    "hate":    ["TrueOffMyChest", "unpopularopinion", "4chan"],
    "irony":   ["memes", "unpopularopinion", "greentext"],
    "neutral": ["AskReddit", "CasualConversation"],
}

# r/4chan + r/greentext = screenshots of 4chan posts, reposted on Reddit.
# (4chan itself is excluded: its robots.txt says "Disallow: /")
IMAGE_SUBREDDITS = ["memes", "dankmemes", "4chan", "greentext"]  # NOT RoastMe (faces)
MAX_IMAGES = 80

session = requests.Session()
session.headers["User-Agent"] = "TP-FR-cyberbullying-student/1.0 (academic project)"


# --------------------------------------------------------------------------
# 2. HTTP helper: polite requests + rate-limit / timeout handling
# --------------------------------------------------------------------------
def api_get(endpoint, params, retries=3):
    """GET an Arctic Shift endpoint. Returns a list of items, or None on failure."""
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        except requests.RequestException as e:
            print(f"   network error ({e}), retry {attempt + 1}")
            time.sleep(5)
            continue

        if r.status_code == 429:                         # rate limited
            wait = int(r.headers.get("X-RateLimit-Reset", 30))
            print(f"   rate limited, waiting {wait}s")
            time.sleep(wait + 1)
            continue

        data = r.json() if r.ok else {"error": f"HTTP {r.status_code}"}
        if data.get("error"):                            # e.g. "Query timed out"
            print(f"   API error: {data['error']} (retry {attempt + 1})")
            time.sleep(3)
            continue

        time.sleep(1)                                    # be polite: ~1 req/s
        return data.get("data", [])
    return None


# --------------------------------------------------------------------------
# 3. Helpers: anonymisation, filtering, row building
# --------------------------------------------------------------------------
def anonymise(username):
    """Irreversible hash: we can group by user without knowing who it is."""
    return hashlib.sha256((SALT + str(username)).encode()).hexdigest()[:12]


def keyword_regex(query):
    """Turn the websearch query into a local regex (used in fallback mode)."""
    terms = [t.strip().strip('"') for t in query.split(" OR ")]
    return re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.I)


def is_valid(text):
    return (
        isinstance(text, str)
        and text not in ("[deleted]", "[removed]")
        and len(text.strip()) >= 15
    )


def make_row(item, group, kind, media=""):
    """Build one row with the structure required by the TP."""
    if kind == "comment":
        text = item.get("body", "")
        post_id = item.get("link_id", "").replace("t3_", "")
        url = f"https://www.reddit.com/r/{item['subreddit']}/comments/{post_id}/_/{item['id']}/"
    else:
        text = (item.get("title", "") + " " + item.get("selftext", "")).strip()
        url = f"https://www.reddit.com/r/{item['subreddit']}/comments/{item['id']}/"

    return {
        "id": f"rd_{item['id']}",
        "date": datetime.fromtimestamp(item["created_utc"], tz=timezone.utc).isoformat(),
        "text": text,
        "media": media,
        "url": url,
        "source": "reddit",
        "metadata": json.dumps({
            "subreddit": item["subreddit"],
            "score": item.get("score"),
            "kind": kind,
            "query_group": group,          # NOT a label! only how we found it
            "author_hash": anonymise(item.get("author")),
        }),
    }


# --------------------------------------------------------------------------
# 4. Comment collection
# --------------------------------------------------------------------------
def collect_comments(subreddit, group, query):
    """Paginate backwards in time until PER_QUERY valid comments are found.

    Strategy: try server-side keyword search first. Busy subreddits often
    time out on keyword search -> fallback: fetch comments and filter locally.
    """
    rows, before = [], DATE_BEFORE
    use_server_search = query is not None
    regex = keyword_regex(query) if query else None

    for _ in range(30):                               # safety cap on pages
        params = {"subreddit": subreddit, "after": DATE_AFTER, "before": before,
                  "limit": 100, "sort": "desc",
                  "fields": "id,author,body,created_utc,subreddit,score,link_id"}
        if use_server_search:
            params["body"] = query

        items = api_get("comments/search", params)
        if items is None and use_server_search:
            print(f"   -> fallback to local filtering for r/{subreddit}")
            use_server_search = False
            continue
        if not items:
            break

        for it in items:
            if not is_valid(it.get("body")) or it.get("author") == "AutoModerator":
                continue
            if regex and not use_server_search and not regex.search(it["body"]):
                continue
            rows.append(make_row(it, group, "comment"))
            if len(rows) >= PER_QUERY:
                return rows
        before = items[-1]["created_utc"]             # next page = older items
    return rows


# --------------------------------------------------------------------------
# 5. Image posts (memes)
# --------------------------------------------------------------------------
def download_image(url, name):
    """Download, verify and save an image as JPEG. Returns local path or ''."""
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")   # also validates
        path = IMG_DIR / f"{name}.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception as e:
        print(f"   image failed ({url}): {e}")
        return ""


def collect_image_posts():
    rows = []
    per_sub = MAX_IMAGES // len(IMAGE_SUBREDDITS)
    for sub in IMAGE_SUBREDDITS:
        print(f"[images] r/{sub}")
        count, before = 0, DATE_BEFORE
        for _ in range(5):                            # up to 5 pages of posts
            params = {"subreddit": sub, "after": DATE_AFTER, "before": before,
                      "over_18": "false", "limit": 100, "sort": "desc",
                      "fields": "id,author,title,selftext,created_utc,subreddit,"
                                "score,url,post_hint,over_18"}
            items = api_get("posts/search", params) or []
            if not items:
                break
            for it in items:
                url = it.get("url", "")
                if it.get("post_hint") != "image" or "i.redd.it" not in url:
                    continue
                path = download_image(url, f"rd_{it['id']}")
                if path:
                    rows.append(make_row(it, "image_post", "post", media=path))
                    count += 1
                if count >= per_sub:
                    break
            if count >= per_sub:
                break
            before = items[-1]["created_utc"]
        print(f"   {count} images")
    return rows


# --------------------------------------------------------------------------
# 6. Main + MLflow tracking
# --------------------------------------------------------------------------
def main():
    import mlflow

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for group, subs in SUBREDDITS.items():
        for sub in subs:
            print(f"[{group}] r/{sub}")
            rows = collect_comments(sub, group, KEYWORDS[group])
            print(f"   collected {len(rows)}")
            all_rows += rows

    all_rows += collect_image_posts()

    df = pd.DataFrame(all_rows).drop_duplicates(subset="id")
    out = DATA_DIR / "reddit_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {out}")

    # ---- MLflow: log HOW the data was produced --------------------------
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP1_Collecte")
    with mlflow.start_run(run_name="scrape_reddit"):
        mlflow.log_params({
            "source": "reddit (Arctic Shift API)",
            "date_after": DATE_AFTER,
            "date_before": DATE_BEFORE,
            "per_query": PER_QUERY,
            "scraped_on": datetime.now(timezone.utc).date().isoformat(),
        })
        mlflow.log_dict({"keywords": KEYWORDS, "subreddits": SUBREDDITS,
                         "image_subreddits": IMAGE_SUBREDDITS}, "collection_config.json")
        mlflow.log_metrics({
            "n_rows": len(df),
            "n_images": int((df["media"] != "").sum()),
        })
        mlflow.log_artifact(str(out), artifact_path="data")


if __name__ == "__main__":
    main()
