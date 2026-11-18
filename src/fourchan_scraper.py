"""
fourchan_scraper.py — TP1 : 4chan collection via the official read-only JSON API
================================================================================
API docs : https://github.com/4chan/4chan-API
API rules we respect:
  - max 1 request / second
  - source disclosed (column `source` + `url` link to 4chan)
  - worksafe boards only (checked automatically with boards.json)

Output: data/fourchan_raw.csv  (same columns as reddit_raw.csv)
    id | date | text | media | url | source | metadata
"""

import html
import json
import random
import re
import time
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# re-use the SAME keywords and helpers as Reddit -> consistent collection
from reddit_scraper import KEYWORDS, anonymise, keyword_regex

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
API = "https://a.4cdn.org"
IMG_CDN = "https://i.4cdn.org"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

BOARDS = ["pol", "int", "v", "sp"]   # hate, hate/ethnic mockery, insults, banter
THREADS_PER_BOARD = 12               # most-replied threads of the catalog
MATCHED_PER_BOARD = 100              # posts that contain a keyword
UNMATCHED_PER_BOARD = 25             # random posts without keyword (diversity)
IMAGES_PER_BOARD = 15
SEED = 42

GROUP_REGEX = {g: keyword_regex(q) for g, q in KEYWORDS.items() if q}

session = requests.Session()
session.headers["User-Agent"] = "TP-FR-cyberbullying-student/1.0 (academic project)"
_last_call = 0.0


# --------------------------------------------------------------------------
# 2. Polite HTTP: never more than 1 request per second (API rule #1)
# --------------------------------------------------------------------------
def get(url, as_json=True):
    """GET a 4chan URL, waiting so we never exceed 1 request per second."""
    global _last_call
    wait = 1.1 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json() if as_json else r.content


def check_robots():
    """Part 1.3 of the TP: verify robots.txt allows the API paths."""
    rp = urllib.robotparser.RobotFileParser(f"{API}/robots.txt")
    try:
        rp.read()
    except Exception:
        print("robots.txt not reachable -> no restriction declared")
        return True
    ok = rp.can_fetch(session.headers["User-Agent"], f"{API}/pol/catalog.json")
    print(f"robots.txt allows API access: {ok}")
    return ok


def check_worksafe(boards):
    """Keep only boards flagged worksafe (ws_board == 1) by 4chan itself."""
    info = {b["board"]: b.get("ws_board", 0) for b in get(f"{API}/boards.json")["boards"]}
    safe = [b for b in boards if info.get(b) == 1]
    for b in set(boards) - set(safe):
        print(f"   /{b}/ is NOT worksafe -> skipped")
    return safe


# --------------------------------------------------------------------------
# 3. Text helpers
# --------------------------------------------------------------------------
def clean_html(com):
    """4chan comments are HTML: convert to plain text, drop >>12345 links."""
    if not com:
        return ""
    text = com.replace("<br>", "\n")
    text = re.sub(r"<[^>]+>", "", text)          # strip tags
    text = html.unescape(text)                   # &gt; -> >, &#039; -> '
    text = re.sub(r">>\d+", "", text)            # reply references
    return text.strip()


def match_group(text):
    """Return the first keyword group found in the text, or None."""
    for g, rx in GROUP_REGEX.items():
        if rx.search(text):
            return g
    return None


def make_row(board, thread_no, post, text, group, n_replies_to_post, media=""):
    """Build one output row (id, date, text, media, url, source, metadata)."""
    return {
        "id": f"4c_{board}_{post['no']}",
        "date": datetime.fromtimestamp(post["time"], tz=timezone.utc).isoformat(),
        "text": text,
        "media": media,
        "url": f"https://boards.4chan.org/{board}/thread/{thread_no}#p{post['no']}",
        "source": "4chan",
        "metadata": json.dumps({
            "board": board,
            "thread": thread_no,
            "is_op": post.get("resto", 0) == 0,
            "replies_received": n_replies_to_post,   # 4chan has no likes:
            "query_group": group or "none",          # engagement = replies
            "poster_hash": anonymise(post["id"]) if "id" in post else None,
        }),
    }


def download_image(board, post):
    """Download a .jpg/.png attachment, verify it, save as JPEG."""
    url = f"{IMG_CDN}/{board}/{post['tim']}{post['ext']}"
    try:
        img = Image.open(BytesIO(get(url, as_json=False))).convert("RGB")
        path = IMG_DIR / f"4c_{board}_{post['no']}.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception as e:
        print(f"   image failed ({url}): {e}")
        return ""


# --------------------------------------------------------------------------
# 4. Board collection
# --------------------------------------------------------------------------
def collect_board(board, rng):
    """Collect keyword-matched posts, random posts and images from one board."""
    print(f"[/{board}/] reading catalog")
    catalog = get(f"{API}/{board}/catalog.json")
    threads = [t for page in catalog for t in page["threads"] if not t.get("sticky")]
    threads = sorted(threads, key=lambda t: t.get("replies", 0), reverse=True)
    threads = threads[:THREADS_PER_BOARD]

    matched, unmatched, image_candidates = [], [], []
    for t in threads:
        try:
            posts = get(f"{API}/{board}/thread/{t['no']}.json")["posts"]
        except requests.HTTPError:
            continue                                  # thread deleted meanwhile

        # engagement: how many times each post is quoted (>>no) in the thread
        quoted = Counter(int(n) for p in posts
                         for n in re.findall(r"&gt;&gt;(\d+)", p.get("com", "")))

        for p in posts:
            text = clean_html(p.get("com", "")) if "capcode" not in p else ""
            if len(text) < 15:                         # skip empty / tiny posts
                continue
            group = match_group(text)
            item = (t["no"], p, text, group, quoted.get(p["no"], 0))
            (matched if group else unmatched).append(item)

            has_img = p.get("ext") in (".jpg", ".png")
            if has_img and not p.get("spoiler") and not p.get("filedeleted"):
                image_candidates.append(item)

    rng.shuffle(matched)
    rng.shuffle(unmatched)
    rng.shuffle(image_candidates)

    rows = [make_row(board, *it) for it in matched[:MATCHED_PER_BOARD]]
    rows += [make_row(board, *it) for it in unmatched[:UNMATCHED_PER_BOARD]]

    # images: attach the downloaded file to the post's row (add row if needed)
    by_id = {r["id"]: r for r in rows}
    n_img = 0
    for thread_no, p, text, group, nq in image_candidates:
        if n_img >= IMAGES_PER_BOARD:
            break
        path = download_image(board, p)
        if not path:
            continue
        n_img += 1
        rid = f"4c_{board}_{p['no']}"
        if rid in by_id:
            by_id[rid]["media"] = path
        else:
            row = make_row(board, thread_no, p, text, group, nq, media=path)
            rows.append(row)
            by_id[rid] = row

    print(f"   {len(rows)} rows ({len(matched)} matched available), {n_img} images")
    return rows


# --------------------------------------------------------------------------
# 5. Main + MLflow
# --------------------------------------------------------------------------
def main():
    """Check robots.txt and worksafe boards, scrape them, save CSV, log to MLflow."""
    import mlflow

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    if not check_robots():
        raise SystemExit("robots.txt disallows access -> stop.")
    boards = check_worksafe(BOARDS)

    rows = []
    for b in boards:
        rows += collect_board(b, rng)

    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    out = DATA_DIR / "fourchan_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {out}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP1_Collecte")
    with mlflow.start_run(run_name="scrape_4chan"):
        mlflow.log_params({
            "source": "4chan (official read-only API)",
            "boards": ",".join(boards),
            "threads_per_board": THREADS_PER_BOARD,
            "matched_per_board": MATCHED_PER_BOARD,
            "unmatched_per_board": UNMATCHED_PER_BOARD,
            "images_per_board": IMAGES_PER_BOARD,
            "seed": SEED,
            "scraped_on": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        })
        mlflow.log_dict({"keywords": KEYWORDS}, "keywords.json")
        mlflow.log_metrics({
            "n_rows": len(df),
            "n_images": int((df["media"] != "").sum()),
        })
        mlflow.log_artifact(str(out), artifact_path="data")


if __name__ == "__main__":
    main()
