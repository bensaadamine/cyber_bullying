"""
image_preprocessing.py — TP2 Part 2 : image preprocessing
=========================================================
For each image: check it opens, remove exact duplicates, convert to RGB JPEG,
resize to 224x224 WITH PADDING (the whole meme is kept, nothing is cropped).
Pixel normalisation (ImageNet mean/std for ResNet50) is done at training time (TP3).
A post whose image is broken keeps its text: only the image is removed.
Output: data/processed/images_224/<id>.jpg  (+ image_path updated in the CSV)
"""

import hashlib

import pandas as pd
from PIL import Image, ImageOps

from utils import PROC, ROOT, log_to_mlflow

SIZE = (224, 224)


def open_image(path):
    """Return the image in RGB, or None if the file is missing or corrupted."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def main():
    """Check, deduplicate, convert and resize every image; update the CSV."""
    csv = PROC / "cyberbullying_processed.csv"
    df = pd.read_csv(csv).fillna({"image_path": "", "image_label": ""})
    out_dir = PROC / "images_224"
    out_dir.mkdir(exist_ok=True)

    stats = {"corrupted": 0, "duplicates": 0}
    seen = set()                                             # fingerprints of images already kept
    for i, row in df[df["image_path"] != ""].iterrows():
        img = open_image(ROOT / row["image_path"])
        fingerprint = hashlib.md5(img.tobytes()).hexdigest() if img else None
        if img is None or fingerprint in seen:
            stats["corrupted" if img is None else "duplicates"] += 1
            df.loc[i, ["image_path", "image_label"]] = ""      # drop the image, keep the text
            continue
        seen.add(fingerprint)
        out = out_dir / f"{row['id']}.jpg"
        ImageOps.pad(img, SIZE, color=(0, 0, 0)).save(out, "JPEG", quality=95)
        df.loc[i, "image_path"] = out.relative_to(ROOT).as_posix()

    stats["valid"] = int((df["image_path"] != "").sum())
    df.to_csv(csv, index=False, encoding="utf-8")
    print(stats)

    log_to_mlflow("TP2_Preparation", "image_preprocessing",
                  params={"size": "224x224", "resize": "padding", "format": "RGB JPEG",
                          "normalisation": "ImageNet mean/std, at training time"},
                  metrics=stats)


if __name__ == "__main__":
    main()
