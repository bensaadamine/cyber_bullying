# Rapport de qualité des données — TP2

## 1. Statistiques descriptives
| Indicateur | Valeur |
|---|---|
| Corpus nettoyé du TP1 | 1510 publications |
| Publications annotées | 600 (Reddit 400, Hateful Memes 200) |
| Images valides | 200 |
| Taux de valeurs manquantes (colonnes essentielles) | 0.0% |
| Doublons supprimés | TP1 : 10 (texte + image) ; TP2 : 0 (exacts et après normalisation) |
| Textes de bots / modérateurs supprimés (TP1) | 13 |

## 2. Distribution des classes
| label | n | part |
|---|---|---|
| non_toxic | 310 | 51.7% |
| irony_provocation | 104 | 17.3% |
| hate_discrimination | 94 | 15.7% |
| insult_mockery | 83 | 13.8% |
| threat_intimidation | 9 | 1.5% |

Ratio de déséquilibre (max / min) : **34**.

## 3. Ensembles train / validation / test
Découpage stratifié et groupé (`StratifiedGroupKFold`) : train 427, validation 86, test 87.

| label | train | validation | test | train_balanced |
|---|---|---|---|---|
| hate_discrimination | 68 | 13 | 13 | 100 |
| insult_mockery | 57 | 11 | 15 | 100 |
| irony_provocation | 75 | 15 | 14 | 100 |
| non_toxic | 221 | 45 | 44 | 150 |
| threat_intimidation | 6 | 2 | 1 | 30 |

## 4. Stratégies et justifications
- **Annotation texte** : 600 publications (200 mèmes + 400 Reddit répartis sur les 9 groupes de requête), guide dans `annotation_guidelines.md`.
- **Annotation image** : labels experts du dataset Hateful Memes (hateful → `hateful_discriminatory`, sinon `neutral`).
- **Prétraitement texte** : depuis le texte brut ; Unicode NFKC, URLs → `<url>`, mentions → `<user>`, emojis convertis en texte, ponctuation et hashtags conservés (ironie), élongations réduites, minuscules.
- **Prétraitement image** : contrôle d'intégrité, doublons exacts (MD5), RGB JPEG, 224×224 avec padding (mème entier conservé) ; normalisation ImageNet appliquée à l'entraînement.
- **Rééquilibrage (train uniquement)** : sous-échantillonnage de `non_toxic` à 150, sur-échantillonnage des minoritaires à 100 (au plus ×5). Validation et test gardent la distribution réelle.

## 5. Contrôles automatiques (✅ tous les contrôles passent)
| dataset | check | passed | detail |
|---|---|---|---|
| processed | check_no_missing | ✅ | 0 missing values in ['id', 'text', 'label', 'annotator_id', 'source', 'group'] |
| processed | check_unique_ids | ✅ | 0 duplicated ids |
| processed | check_valid_labels | ✅ | unknown labels: none |
| processed | check_images_exist | ✅ | 0 missing files, 0 images without label |
| processed | check_no_duplicates | ✅ | 0 duplicated (text, image) pairs |
| train | check_no_missing | ✅ | 0 missing values in ['id', 'text', 'label', 'annotator_id', 'source', 'group'] |
| train | check_unique_ids | ✅ | 0 duplicated ids |
| train | check_valid_labels | ✅ | unknown labels: none |
| train | check_images_exist | ✅ | 0 missing files, 0 images without label |
| train | check_no_duplicates | ✅ | 0 duplicated (text, image) pairs |
| validation | check_no_missing | ✅ | 0 missing values in ['id', 'text', 'label', 'annotator_id', 'source', 'group'] |
| validation | check_unique_ids | ✅ | 0 duplicated ids |
| validation | check_valid_labels | ✅ | unknown labels: none |
| validation | check_images_exist | ✅ | 0 missing files, 0 images without label |
| validation | check_no_duplicates | ✅ | 0 duplicated (text, image) pairs |
| test | check_no_missing | ✅ | 0 missing values in ['id', 'text', 'label', 'annotator_id', 'source', 'group'] |
| test | check_unique_ids | ✅ | 0 duplicated ids |
| test | check_valid_labels | ✅ | unknown labels: none |
| test | check_images_exist | ✅ | 0 missing files, 0 images without label |
| test | check_no_duplicates | ✅ | 0 duplicated (text, image) pairs |
| splits | check_no_leakage | ✅ | no overlap |
| splits | check_class_proportions | ✅ | max gap 3.4% (test/insult_mockery) |

## 6. Principales difficultés
- **Classe menace quasi absente** (9 exemples) : les vraies menaces sont supprimées par la modération Reddit ; les victimes qui les citent sont annotées `non_toxic`.
- **Biais de source** : 88 des 94 publications haineuses sont des mèmes → risque que le modèle apprenne « mème = haine ».
- **Biais de longueur** : `non_toxic` ≈ 80 mots contre ≈ 13 pour `hate_discrimination`.
- **Annotation** : texte annoté par un seul annotateur (modèle de langage) ; images limitées à 2 des 4 étiquettes.
- **Peu de données** : 600 exemples pour du Deep Learning (TP3) → privilégier le transfert d'apprentissage.
