"""
reddit_scraper.py — TP1 : collect Reddit comments
=================================================
Source: Arctic Shift, a free public archive of Reddit (no API key needed;
Reddit stopped giving new API keys on 2026-10-31).
Output: data/reddit_raw.csv   (id | date | text | media | url | source | metadata)
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from utils import DATA, log_to_mlflow

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
DATE_AFTER, DATE_BEFORE = "2026-05-01", "2026-11-01"
PER_QUERY = 60                     # comments kept per (group, subreddit)
SALT = "tp-fr-2026"                # used to anonymise usernames

# (group, subreddits, keywords)   keywords=None -> no keyword filter
# "query_group" only says HOW a comment was found, it is NOT its label.
QUERIES = [
    ("insult", ["RoastMe", "gaming", "4chan"],
     'ugly OR loser OR stupid OR idiot OR pathetic OR fat'),
    ("threat", ["bullying", "TrueOffMyChest", "4chan"],
     '"kill yourself" OR kys OR "watch your back" OR "find you" OR "hurt you"'),
    ("hate", ["TrueOffMyChest", "unpopularopinion", "4chan"],
     'subhuman OR "go back to" OR disgusting OR vermin OR worthless'),
    ("irony", ["memes", "unpopularopinion", "greentext"],
     'cope OR seethe OR ratio OR "nobody asked" OR "ok boomer"'),
    ("neutral", ["AskReddit", "CasualConversation"], None),
    ("roast", ["RoastMe"], None),  # every top-level RoastMe comment is a roast
    ("hate_immigration", ["europe", "unitedkingdom", "immigration"],
     'invaders OR illegals OR parasites OR "deport them" OR "third world" OR '
     '"go back to your country" OR "send them back"'),
    ("provocation_sports", ["soccer", "nba"],
     'cope OR seethe OR clown OR clowns OR bottlers OR "cry more" OR "stay mad" OR ratio'),
    ("threat_quoted", ["legaladvice", "relationship_advice"],
     '"kill you" OR "kill me" OR "know where you live" OR "knows where i live" OR '
     '"threatened to" OR "you\'re dead" OR "watch your back"'),
]
ROAST_OPTIONS = {"n": 200, "min_score": 5, "top_level": True}


def get_page(params):
    """Ask the API for one page (max 100 comments); retry on errors, None if it keeps failing."""
    for _ in range(3):
        try:
            r = requests.get(API, params=params, timeout=60)
            if r.status_code == 429:               # too many requests: wait, then retry
                time.sleep(30)
                continue
            data = r.json()
            if not data.get("error"):
                time.sleep(1)                      # be polite: about 1 request per second
                return data["data"]
        except requests.RequestException:
            pass
        time.sleep(5)
    return None


def make_row(c, group):
    """Turn one API comment into one row of our table (username replaced by a hash)."""
    post_id = c["link_id"].replace("t3_", "")
    return {
        "id": f"rd_{c['id']}",
        "date": datetime.fromtimestamp(c["created_utc"], tz=timezone.utc).isoformat(),
        "text": c["body"],
        "media": "",
        "url": f"https://www.reddit.com/r/{c['subreddit']}/comments/{post_id}/_/{c['id']}/",
        "source": "reddit",
        "metadata": json.dumps({
            "subreddit": c["subreddit"],
            "score": c.get("score"),
            "kind": "comment",
            "query_group": group,
            "author_hash": hashlib.sha256((SALT + str(c.get("author"))).encode()).hexdigest()[:12],
        }),
    }


def collect(subreddit, group, keywords, n=PER_QUERY, min_score=None, top_level=False):
    """Collect up to n valid comments, going back in time one page at a time."""
    words = [k.strip().strip('"') for k in keywords.split(" OR ")] if keywords else []
    regex = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I) if words else None
    server_search = keywords is not None
    rows, before = [], DATE_BEFORE

    for _ in range(30):                                       # at most 30 pages
        params = {"subreddit": subreddit, "after": DATE_AFTER, "before": before,
                  "limit": 100, "sort": "desc"}
        if server_search:
            params["body"] = keywords
        if top_level:
            params["parent_id"] = ""                          # "" = direct replies to the post
        page = get_page(params)

        if page is None and server_search:    # keyword search timed out (busy subreddit):
            server_search = False             # fetch everything and filter it here instead
            continue
        if not page:
            break

        for c in page:
            text = c.get("body", "")
            if text in ("[deleted]", "[removed]") or len(text.strip()) < 15:
                continue
            if c.get("author") == "AutoModerator":
                continue
            if regex and not server_search and not regex.search(text):
                continue
            if min_score and (c.get("score") or 0) < min_score:
                continue
            rows.append(make_row(c, group))
            if len(rows) >= n:
                return rows
        before = page[-1]["created_utc"]                      # next page = older comments
    return rows


def main():
    """Run every query, save data/reddit_raw.csv and log the collection to MLflow."""
    rows = []
    for group, subreddits, keywords in QUERIES:
        options = ROAST_OPTIONS if group == "roast" else {}
        for sub in subreddits:
            found = collect(sub, group, keywords, **options)
            print(f"[{group}] r/{sub}: {len(found)}")
            rows += found

    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    out = DATA / "reddit_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {out}")

    log_to_mlflow("TP1_Collecte", "scrape_reddit",
                  params={"source": "reddit (Arctic Shift)", "date_after": DATE_AFTER,
                          "date_before": DATE_BEFORE, "per_query": PER_QUERY,
                          "scraped_on": datetime.now(timezone.utc).date().isoformat(),
                          **{f"keywords_{g}": k or "none" for g, _, k in QUERIES}},
                  metrics={"n_rows": len(df)},
                  files=[out])


if __name__ == "__main__":
    main()
