"""
hateful_memes_sampler.py — TP1 : 2nd source = Hateful Memes (text + image)
=========================================================================
Dataset: Meta Hateful Memes (Kiela et al. 2020), Hugging Face mirror.
License: non-commercial research -> cite it, never republish the images.
The full dataset is ~8 GB: "streaming" reads it row by row, so we only
download what we keep. Original label: 1 = hateful, 0 = not hateful.
Output: data/hateful_memes_raw.csv + data/images/hm_<id>.jpg
"""

import json

import pandas as pd
from datasets import load_dataset

from utils import DATA, ROOT, log_to_mlflow

DATASET = "cs5242-hateful-memes/hateful-memes-data"
N_PER_LABEL = 100              # 100 hateful + 100 not hateful
SEED = 42


def main():
    """Stream the dataset, keep a balanced sample, save the images and the CSV."""
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    stream = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=SEED, buffer_size=2000)

    left = {0: N_PER_LABEL, 1: N_PER_LABEL}          # how many we still need per label
    rows = []
    for ex in stream:
        label = int(ex["label"])
        if left.get(label, 0) == 0:
            continue
        path = DATA / "images" / f"hm_{ex['id']}.jpg"
        img = ex["image"].convert("RGB")
        img.thumbnail((512, 512))                     # smaller files, enough for 224x224 later
        img.save(path, "JPEG", quality=90)
        rows.append({
            "id": f"hm_{ex['id']}",
            "date": "",                               # memes have no date
            "text": ex["text"],                       # caption written on the meme
            "media": path.relative_to(ROOT).as_posix(),
            "url": f"https://huggingface.co/datasets/{DATASET}",
            "source": "hateful_memes",
            "metadata": json.dumps({"original_label": label, "query_group": "meme"}),
        })
        left[label] -= 1
        if sum(left.values()) == 0:
            break

    df = pd.DataFrame(rows)
    out = DATA / "hateful_memes_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} memes -> {out}")

    log_to_mlflow("TP1_Collecte", "sample_hateful_memes",
                  params={"dataset": DATASET, "n_per_label": N_PER_LABEL, "seed": SEED,
                          "license": "Meta Hateful Memes (non-commercial research)"},
                  metrics={"n_memes": len(df)},
                  files=[out])


if __name__ == "__main__":
    main()
