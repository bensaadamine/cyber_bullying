"""
data_validation.py — TP2 Parts 5 & 6 : automatic checks + quality report + MLflow summary
=========================================================================================
Each check_* function returns (passed: bool, detail: str).
Run:  python src/data_validation.py   -> exit code 1 if any check fails
Output: reports/data_quality_report.md
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
REPORT = ROOT / "reports" / "data_quality_report.md"

LABELS = {"non_toxic", "insult_mockery", "hate_discrimination",
          "threat_intimidation", "irony_provocation"}
IMAGE_LABELS = {"neutral", "insulting_humiliating", "hateful_discriminatory", "ambiguous_ironic"}
ESSENTIAL = ["id", "text", "label", "annotator_id", "source", "group"]
SPLITS = ["train", "validation", "test"]
TOLERANCE = 0.05   # max gap between a class share in a split and in the full dataset


# --------------------------------------------------------------------------
# 1. Checks
# --------------------------------------------------------------------------
def check_no_missing(df, cols=ESSENTIAL):
    """No empty value in the essential columns."""
    n = int(df[cols].isna().sum().sum())
    return n == 0, f"{n} missing values in {cols}"


def check_unique_ids(df):
    """Every id appears only once."""
    n = int(df["id"].duplicated().sum())
    return n == 0, f"{n} duplicated ids"


def check_valid_labels(df):
    """Text labels and image labels belong to the taxonomy."""
    bad = set(df["label"]) - LABELS
    img = set(df["image_label"].dropna()) - IMAGE_LABELS - {""}
    return not bad and not img, f"unknown labels: {sorted(bad | img) or 'none'}"


def check_images_exist(df):
    """Every image path points to an existing file, and only memes have images."""
    paths = df["image_path"].dropna()
    paths = paths[paths != ""]
    missing = [p for p in paths if not (ROOT / p).exists()]
    no_label = int(((df["image_path"].fillna("") != "") & (df["image_label"].fillna("") == "")).sum())
    return not missing and no_label == 0, f"{len(missing)} missing files, {no_label} images without label"


def check_no_duplicates(df):
    """No identical (text, image) pair."""
    n = int(df.duplicated(["text", "image_path"]).sum())
    return n == 0, f"{n} duplicated (text, image) pairs"


def check_no_leakage(splits):
    """No group (thread / meme caption) and no text shared between two splits."""
    issues = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = set(splits[a]["group"]) & set(splits[b]["group"])
            t = set(splits[a]["text"]) & set(splits[b]["text"])
            if g or t:
                issues.append(f"{a}/{b}: {len(g)} groups, {len(t)} texts")
    return not issues, "; ".join(issues) or "no overlap"


def check_class_proportions(full, splits, tol=TOLERANCE):
    """Each class share in each split stays within ±tol of the full dataset."""
    ref = full["label"].value_counts(normalize=True)
    gaps = {f"{name}/{c}": abs(part["label"].value_counts(normalize=True).get(c, 0) - p)
            for name, part in splits.items() for c, p in ref.items()}
    worst = max(gaps, key=gaps.get)
    return gaps[worst] <= tol, f"max gap {gaps[worst]:.1%} ({worst})"


def run_checks(full, splits):
    """Run every check on the full dataset and on each split; return a result table."""
    rows = []
    for name, df in {"processed": full, **splits}.items():
        for check in (check_no_missing, check_unique_ids, check_valid_labels,
                      check_images_exist, check_no_duplicates):
            ok, detail = check(df)
            rows.append((name, check.__name__, ok, detail))
    for check, args in ((check_no_leakage, (splits,)), (check_class_proportions, (full, splits))):
        ok, detail = check(*args)
        rows.append(("splits", check.__name__, ok, detail))
    return pd.DataFrame(rows, columns=["dataset", "check", "passed", "detail"])


# --------------------------------------------------------------------------
# 2. Quality report (Markdown)
# --------------------------------------------------------------------------
def md_table(df, index=True):
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build_report(full, splits, balanced, results):
    """Write the data quality report with the numbers computed from the data."""
    raw = pd.read_csv(DATA / "cyberbullying_raw_clean.csv")
    n_img = int((full["image_path"].fillna("") != "").sum())
    missing_rate = full[ESSENTIAL].isna().mean().mean()
    per_class = full["label"].value_counts()
    dist = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()}).fillna(0).astype(int)
    dist["train_balanced"] = balanced["label"].value_counts()
    status = "✅ tous les contrôles passent" if results["passed"].all() else "❌ des contrôles échouent"

    md = f"""# Rapport de qualité des données — TP2

## 1. Statistiques descriptives
| Indicateur | Valeur |
|---|---|
| Corpus nettoyé du TP1 | {len(raw)} publications |
| Publications annotées | {len(full)} (Reddit {int((full['source'] == 'reddit').sum())}, Hateful Memes {int((full['source'] == 'hateful_memes').sum())}) |
| Images valides | {n_img} |
| Taux de valeurs manquantes (colonnes essentielles) | {missing_rate:.1%} |
| Doublons supprimés | TP1 : 10 (texte + image) ; TP2 : 0 (exacts et après normalisation) |
| Textes de bots / modérateurs supprimés (TP1) | 13 |

## 2. Distribution des classes
{md_table(per_class.rename('n').to_frame().assign(part=lambda d: (d['n'] / d['n'].sum()).map('{:.1%}'.format)))}

Ratio de déséquilibre (max / min) : **{per_class.max() / per_class.min():.0f}**.

## 3. Ensembles train / validation / test
Découpage stratifié et groupé (`StratifiedGroupKFold`) : {', '.join(f'{k} {len(v)}' for k, v in splits.items())}.

{md_table(dist)}

## 4. Stratégies et justifications
- **Annotation texte** : 600 publications (200 mèmes + 400 Reddit répartis sur les 9 groupes de requête), guide dans `annotation_guidelines.md`.
- **Annotation image** : labels experts du dataset Hateful Memes (hateful → `hateful_discriminatory`, sinon `neutral`).
- **Prétraitement texte** : depuis le texte brut ; Unicode NFKC, URLs → `<url>`, mentions → `<user>`, emojis convertis en texte, ponctuation et hashtags conservés (ironie), élongations réduites, minuscules.
- **Prétraitement image** : contrôle d'intégrité, doublons exacts (MD5), RGB JPEG, 224×224 avec padding (mème entier conservé) ; normalisation ImageNet appliquée à l'entraînement.
- **Rééquilibrage (train uniquement)** : sous-échantillonnage de `non_toxic` à 150, sur-échantillonnage des minoritaires à 100 (au plus ×5). Validation et test gardent la distribution réelle.

## 5. Contrôles automatiques ({status})
{md_table(results.assign(passed=results['passed'].map({True: '✅', False: '❌'})), index=False)}

## 6. Principales difficultés
- **Classe menace quasi absente** (9 exemples) : les vraies menaces sont supprimées par la modération Reddit ; les victimes qui les citent sont annotées `non_toxic`.
- **Biais de source** : 88 des 94 publications haineuses sont des mèmes → risque que le modèle apprenne « mème = haine ».
- **Biais de longueur** : `non_toxic` ≈ 80 mots contre ≈ 13 pour `hate_discrimination`.
- **Annotation** : texte annoté par un seul annotateur (modèle de langage) ; images limitées à 2 des 4 étiquettes.
- **Peu de données** : 600 exemples pour du Deep Learning (TP3) → privilégier le transfert d'apprentissage.
"""
    REPORT.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------
# 3. Main + MLflow summary (Part 6)
# --------------------------------------------------------------------------
def main():
    """Run the checks, write the report, log the dataset summary to MLflow."""
    import mlflow

    full = pd.read_csv(PROC / "cyberbullying_processed.csv")
    splits = {s: pd.read_csv(PROC / f"{s}.csv") for s in SPLITS}
    balanced = pd.read_csv(PROC / "train_balanced.csv")

    results = run_checks(full, splits)
    print(results.to_string(index=False))
    build_report(full, splits, balanced, results)
    print(f"\nReport -> {REPORT}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("TP2_Preparation")
    with mlflow.start_run(run_name="dataset_summary"):
        mlflow.log_params({
            "text_preprocessing": "NFKC, <url>, <user>, demojize, keep punctuation, lowercase",
            "image_preprocessing": "integrity, md5 dedup, RGB, 224x224 pad",
            "rebalancing": "under non_toxic->150, over minorities->100 (max x5), train only",
            "split": "StratifiedGroupKFold ~70/15/15",
        })
        mlflow.log_metrics({
            "n_annotated": len(full),
            "n_images": int((full["image_path"].fillna("") != "").sum()),
            **{f"n_{k}": len(v) for k, v in splits.items()},
            "n_train_balanced": len(balanced),
            **{f"share_{c}": float(p) for c, p in full["label"].value_counts(normalize=True).items()},
            "checks_passed": int(results["passed"].sum()),
            "checks_total": len(results),
        })
        mlflow.log_artifact(str(REPORT))
        for f in ["cyberbullying_processed.csv", "train.csv", "validation.csv", "test.csv", "train_balanced.csv"]:
            mlflow.log_artifact(str(PROC / f), "data")

    sys.exit(0 if results["passed"].all() else 1)


if __name__ == "__main__":
    main()
