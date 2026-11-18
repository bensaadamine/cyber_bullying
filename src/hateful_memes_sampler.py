"""
hateful_memes_sampler.py — TP1 : 2nd source = Hateful Memes (Meta, Kiela et al. 2020)
=====================================================================================
Dataset : https://huggingface.co/datasets/cs5242-hateful-memes/hateful-memes-data
          (mirror of the Facebook Hateful Memes Challenge — non-commercial research
           license: cite it, do NOT republish the images publicly)

The full dataset is ~8 GB. We never download it: `streaming=True` reads the
Parquet shards row by row, and we stop as soon as our quota is reached.

Original labels: 1 = hateful, 0 = not hateful.
We keep a BALANCED sample: N_PER_LABEL of each.

Output: data/hateful_memes_raw.csv  (same columns as reddit_raw.csv)
        data/images/hm_<id>.jpg
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

DATASET = "cs5242-hateful-memes/hateful-memes-data"
SPLIT = "train"
N_PER_LABEL = 100            # 100 hateful + 100 not hateful = 200 memes
SEED = 42
SHUFFLE_BUFFER = 2000        # streaming shuffle = random pick inside a 2000-row window
MAX_SIDE = 512               # downscale big images (TP2 will resize to 224 anyway)


def main():
    """Stream the dataset, keep a balanced sample of memes, save images + CSV, log to MLflow."""
    import mlflow

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. open the dataset as a STREAM (nothing downloaded up front)
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    # 2. take rows until each label has N_PER_LABEL examples
    quota = {0: N_PER_LABEL, 1: N_PER_LABEL}
    rows, seen = [], 0
    for ex in ds:
        seen += 1
        label = int(ex["label"])
        if label not in quota or quota[label] == 0:
            continue

        img = ex["image"].convert("RGB")               # PIL image, decoded by `datasets`
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        name = f"hm_{ex['id']}"
        path = IMG_DIR / f"{name}.jpg"
        img.save(path, "JPEG", quality=90)

        rows.append({
            "id": name,
            "date": "",                                # memes have no publication date
            "text": ex["text"],                        # caption written on the meme
            "media": str(path.relative_to(ROOT)).replace("\\", "/"),
            "url": f"https://huggingface.co/datasets/{DATASET}",
            "source": "hateful_memes",
            "metadata": json.dumps({
                "dataset_split": SPLIT,
                "original_label": label,               # 1 hateful / 0 not hateful
                "query_group": "meme",
            }),
        })
        quota[label] -= 1
        if sum(quota.values()) == 0:
            break

    df = pd.DataFrame(rows)
    out = DATA_DIR / "hateful_memes_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    n_hate = sum(json.loads(m)["original_label"] == 1 for m in df["metadata"])
    print(f"Read {seen} rows of the stream -> kept {len(df)} memes "
          f"({n_hate} hateful / {len(df) - n_hate} not hateful)")
    print(f"Saved -> {out}")

    # 3. MLflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP1_Collecte")
    with mlflow.start_run(run_name="sample_hateful_memes"):
        mlflow.log_params({
            "source": DATASET, "split": SPLIT, "n_per_label": N_PER_LABEL,
            "seed": SEED, "shuffle_buffer": SHUFFLE_BUFFER, "max_side": MAX_SIDE,
            "license": "Meta Hateful Memes (non-commercial research)",
            "sampled_on": datetime.now(timezone.utc).date().isoformat(),
        })
        mlflow.log_metrics({"n_memes": len(df), "n_hateful": n_hate,
                            "rows_streamed": seen})
        mlflow.log_artifact(str(out), artifact_path="data")


if __name__ == "__main__":
    main()
