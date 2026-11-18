"""
data_validation.py — TP2 Parts 5 & 6 : automatic checks + quality report
=======================================================================
Each check returns True (OK) or False (problem).
Output: reports/data_quality_report.md  + an MLflow run "dataset_summary".
The script stops with an error code if a check fails.
"""

import sys

import pandas as pd

from utils import DATA, PROC, REPORTS, ROOT, log_to_mlflow

LABELS = {"non_toxic", "insult_mockery", "hate_discrimination", "threat_intimidation", "irony_provocation"}
ESSENTIAL = ["id", "text", "label", "annotator_id", "source", "group"]
SPLITS = ["train", "validation", "test"]


# ---- the checks ------------------------------------------------------------
def no_missing_values(df):
    """No empty value in the essential columns."""
    return df[ESSENTIAL].notna().all().all()


def unique_ids(df):
    """Every id appears only once."""
    return df["id"].is_unique


def valid_labels(df):
    """Every label belongs to our 5 classes."""
    return set(df["label"]) <= LABELS


def images_exist(df):
    """Every image path points to a real file."""
    paths = df["image_path"].dropna()
    return all((ROOT / p).exists() for p in paths if p)


def no_duplicates(df):
    """No identical (text, image) pair."""
    return not df.duplicated(["text", "image_path"]).any()


def no_leakage(splits):
    """No group (thread / meme caption) shared between two splits."""
    groups = [set(s["group"]) for s in splits.values()]
    return not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def same_class_proportions(full, splits, tolerance=0.05):
    """Each class share in each split is within ±5 points of the full dataset."""
    ref = full["label"].value_counts(normalize=True)
    for s in splits.values():
        share = s["label"].value_counts(normalize=True).reindex(ref.index, fill_value=0)
        if (share - ref).abs().max() > tolerance:
            return False
    return True


def main():
    """Run the checks, write the quality report, log the dataset summary to MLflow."""
    full = pd.read_csv(PROC / "cyberbullying_processed.csv")
    splits = {s: pd.read_csv(PROC / f"{s}.csv") for s in SPLITS}
    balanced = pd.read_csv(PROC / "train_balanced.csv")

    results = {}
    for name, df in {"processed": full, **splits}.items():
        for check in (no_missing_values, unique_ids, valid_labels, images_exist, no_duplicates):
            results[f"{name}: {check.__name__}"] = bool(check(df))
    results["splits: no_leakage"] = no_leakage(splits)
    results["splits: same_class_proportions"] = same_class_proportions(full, splits)
    for name, ok in results.items():
        print("OK  " if ok else "FAIL", name)

    # ---- quality report ----------------------------------------------------
    n_images = int(full["image_path"].notna().sum())
    classes = full["label"].value_counts()
    table = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()})
    table["train_balanced"] = balanced["label"].value_counts()
    checks = "\n".join(f"- {'✅' if ok else '❌'} {name}" for name, ok in results.items())

    report = f"""# Rapport de qualité des données — TP2

## 1. Statistiques
- Corpus nettoyé du TP1 : {len(pd.read_csv(DATA / 'cyberbullying_raw_clean.csv'))} publications
- Publications annotées : {len(full)} (Reddit {(full['source'] == 'reddit').sum()}, Hateful Memes {(full['source'] == 'hateful_memes').sum()})
- Images valides : {n_images}
- Valeurs manquantes (colonnes essentielles) : {full[ESSENTIAL].isna().mean().mean():.1%}
- Doublons supprimés : 10 au TP1 (texte + image), 0 au TP2
- Messages de bots / modérateurs supprimés (TP1) : 13

## 2. Distribution des classes (ratio max / min = {classes.max() / classes.min():.0f})
```
{(classes.to_frame('n').assign(part=(classes / len(full)).map('{:.1%}'.format))).to_string()}
```

## 3. Ensembles train / validation / test (+ train rééquilibré)
```
{table.fillna(0).astype(int).to_string()}
```

## 4. Stratégies
- Annotation texte : 600 publications (200 mèmes + 400 Reddit répartis sur les groupes de requête), guide `annotation_guidelines.md`.
- Annotation image : labels experts du dataset Hateful Memes (hateful → `hateful_discriminatory`, sinon `neutral`).
- Prétraitement texte : depuis le texte brut ; URLs → `<url>`, mentions → `<user>`, emojis en texte, ponctuation et hashtags conservés, minuscules.
- Prétraitement image : contrôle d'intégrité, doublons exacts, RGB, 224×224 avec padding ; normalisation ImageNet à l'entraînement.
- Découpage : `StratifiedGroupKFold` ≈ 70/15/15, groupes = fil Reddit / légende de mème (pas de fuite).
- Rééquilibrage (train uniquement) : `non_toxic` → 150, autres classes → 100 (au plus ×5).

## 5. Contrôles automatiques
{checks}

## 6. Principales difficultés
- Classe menace quasi absente (9 exemples) : les vraies menaces sont supprimées par la modération Reddit.
- Biais de source : presque toute la haine vient des mèmes → risque « mème = haine ».
- Biais de longueur : `non_toxic` ≈ 80 mots contre ≈ 13 pour `hate_discrimination`.
- Un seul annotateur pour le texte (modèle de langage) ; 2 étiquettes image sur 4.
- Peu de données (600) pour du Deep Learning → transfert d'apprentissage au TP3.
"""
    out = REPORTS / "data_quality_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nReport -> {out}")

    log_to_mlflow("TP2_Preparation", "dataset_summary",
                  params={"rebalancing": "non_toxic->150, others->100 (max x5), train only",
                          "split": "StratifiedGroupKFold ~70/15/15"},
                  metrics={"n_annotated": len(full), "n_images": n_images,
                           **{f"n_{k}": len(v) for k, v in splits.items()},
                           "n_train_balanced": len(balanced),
                           **{f"share_{c}": float(p) for c, p in (classes / len(full)).items()},
                           "checks_passed": sum(results.values())},
                  files=[out] + [PROC / f"{s}.csv" for s in SPLITS + ["train_balanced"]])
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
