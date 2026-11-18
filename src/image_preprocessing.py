"""
image_preprocessing.py — TP2 Part 2 (images)
============================================
For every post with an image:
  1. integrity check (file exists, PIL can decode it)   -> corrupted = image dropped
  2. exact duplicates (same pixels, MD5)                 -> duplicate = image dropped
     near duplicates (perceptual hash, distance <= 4)    -> only REPORTED: in Hateful
     Memes the same picture with a different caption is a different meme
  3. common format: RGB JPEG
  4. common size 224 x 224 with padding (keeps the whole meme, no distortion,
     no cropped text)
  5. pixel normalisation: NOT stored on disk; applied at training time (TP3) with the
     ImageNet mean/std expected by ResNet50 (logged below)

The post itself is always kept (its text stays usable): only the image is removed.

Input / output: data/processed/cyberbullying_processed.csv (image_path updated)
Images        : data/processed/images_224/<id>.jpg
"""

import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "processed" / "cyberbullying_processed.csv"
OUT_DIR = ROOT / "data" / "processed" / "images_224"

SIZE = (224, 224)
NEAR_DUP_DISTANCE = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_valid(path):
    """Return an RGB image, or None if the file is missing / corrupted."""
    try:
        with Image.open(path) as im:
            im.verify()                       # detects truncated / broken files
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def main():
    """Check, deduplicate, convert and resize every image; update the CSV; log to MLflow."""
    import mlflow

    df = pd.read_csv(CSV)
    df["image_path"] = df["image_path"].fillna("")
    df["image_label"] = df["image_label"].fillna("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"n_images": 0, "corrupted": 0, "exact_duplicates": 0, "near_duplicates": 0}
    seen_md5, seen_phash = {}, []

    for i, row in df[df["image_path"] != ""].iterrows():
        stats["n_images"] += 1
        img = load_valid(ROOT / row["image_path"])
        if img is None:
            stats["corrupted"] += 1
            df.loc[i, ["image_path", "image_label"]] = ""
            continue

        md5 = hashlib.md5(img.tobytes()).hexdigest()
        if md5 in seen_md5:
            stats["exact_duplicates"] += 1
            df.loc[i, ["image_path", "image_label"]] = ""
            continue
        seen_md5[md5] = row["id"]

        ph = imagehash.phash(img)
        if any(ph - other <= NEAR_DUP_DISTANCE for other in seen_phash):
            stats["near_duplicates"] += 1
        seen_phash.append(ph)

        out = OUT_DIR / f"{row['id']}.jpg"
        ImageOps.pad(img, SIZE, color=(0, 0, 0)).save(out, "JPEG", quality=95)
        df.loc[i, "image_path"] = str(out.relative_to(ROOT)).replace("\\", "/")

    stats["n_valid"] = int((df["image_path"] != "").sum())
    df.to_csv(CSV, index=False, encoding="utf-8")
    for k, v in stats.items():
        print(f"{k:<18} {v}")
    print(f"\nImages saved -> {OUT_DIR}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP2_Preparation")
    with mlflow.start_run(run_name="image_preprocessing"):
        mlflow.log_params({"size": "224x224", "resize": "pad (keep aspect ratio)",
                           "format": "RGB JPEG q95", "near_dup_distance": NEAR_DUP_DISTANCE,
                           "normalisation": "at training time (ImageNet)",
                           "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD})
        mlflow.log_metrics(stats)


if __name__ == "__main__":
    main()
