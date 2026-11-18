# TP1 — Collecte et exploration des données
**Projet fil rouge MLOps : Cyberbullying Detection sur données multimodales (texte + image)**
FST — IDS 5 — 2026-2027

Constituer un corpus multimodal (texte + images) de messages potentiellement liés au
cyberharcèlement, nettoyé, documenté et traçable (MLflow), qui servira de base au TP2.

---

## 1. Structure du projet

```
TP_FR/
├── data/
│   ├── reddit_raw.csv                 # sortie brute du scraper
│   ├── cyberbullying_raw.csv          # corpus brut (toutes les métadonnées)
│   ├── cyberbullying_raw_clean.csv    # corpus nettoyé
│   └── images/                        # 80 images (rd_<id>.jpg)
├── notebooks/
│   └── TP1_collecte.ipynb             # Partie 4 : analyse + mini-rapport
├── src/
│   ├── reddit_scraper.py              # Partie 2 : extraction Reddit
│   ├── fourchan_scraper.py            # non utilisé (robots.txt, voir §3)
│   └── cleaning.py                    # Partie 3 : nettoyage
├── reports/tp1_distribution.png       # figures
├── mini_report.md                     # Partie 4.3 (généré par le notebook)
├── mlflow.db / mlruns/                # Partie 6 : suivi MLflow
├── requirements.txt
└── README.md
```

`src/` et `reports/` s'ajoutent à la structure demandée : le code de collecte est
réutilisable (scripts) et le notebook se limite à l'analyse.

## 2. Reproduire

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/reddit_scraper.py      # ~10 min -> data/reddit_raw.csv + data/images/
python src/cleaning.py            # -> cyberbullying_raw.csv + cyberbullying_raw_clean.csv
jupyter notebook notebooks/TP1_collecte.ipynb   # Run All -> mini_report.md
mlflow ui                         # expérience « TP1_Collecte »
```

La fenêtre temporelle est fixe (passé) et les tirages aléatoires utilisent `seed=42` :
une nouvelle exécution redonne le même corpus (aux suppressions Reddit près).

## 3. Choix des sources

| Source | Décision | Raison |
|---|---|---|
| **Reddit** (archive Arctic Shift) | ✅ retenue | Gratuit, sans clé, données publiques, score de votes. L'API officielle (PRAW) n'accepte plus de nouvelles clés depuis le 31/10/2026 ; Arctic Shift est le successeur de Pushshift. |
| X / Twitter | ❌ | Plus de lecture gratuite (API à l'usage), snscrape et Nitter hors service. |
| 4chan (API officielle) | ❌ | `robots.txt` de `a.4cdn.org` : `Disallow: /` → respecté. Le scraper (`fourchan_scraper.py`) vérifie le robots.txt et s'arrête ; il est conservé comme preuve de la vérification. Le contenu 4chan est présent **indirectement** via r/4chan et r/greentext (captures repostées sur Reddit). |
| YouTube | ❌ (choix) | Commentaires jugés peu représentatifs du harcèlement. |

⚠️ **Limite** : une seule source effective (Reddit) alors que le TP en demande deux —
choix à valider avec l'enseignante.

## 4. Périmètre de collecte

- **Fenêtre** : 01/05/2026 → 01/11/2026 (messages de plus de 36 h : scores définitifs).
- **Volume** : 60 commentaires max par couple (subreddit, groupe) + 20 images par subreddit d'images.

| Groupe de requête | Mots-clés (recherche plein texte) | Subreddits |
|---|---|---|
| insult | ugly, loser, stupid, idiot, pathetic, fat | RoastMe, gaming, 4chan |
| threat | "kill yourself", kys, "watch your back", "find you", "hurt you" | bullying, TrueOffMyChest, 4chan |
| hate | subhuman, "go back to", disgusting, vermin, worthless | TrueOffMyChest, unpopularopinion, 4chan |
| irony | cope, seethe, ratio, "nobody asked", "ok boomer" | memes, unpopularopinion, greentext |
| neutral | *(aucun — échantillon aléatoire)* | AskReddit, CasualConversation |
| images | posts `post_hint=image`, `over_18=false` | memes, dankmemes, 4chan, greentext |

Le `query_group` indique **comment** un message a été trouvé, **pas** son étiquette
(l'annotation est faite au TP2).

## 5. Légalité et éthique

- Données publiques uniquement, via une API documentée ; ~1 requête/s (gestion du code 429).
- `robots.txt` vérifié (4chan exclu pour cette raison).
- **Aucune donnée personnelle** : pseudonymes remplacés par un hash SHA-256 salé
  (`author_hash`), mentions `u/…` et `@…` supprimées du texte.
- Contenu NSFW exclu à la collecte (`over_18=false`), r/RoastMe exclu des images (visages).
- Images : vérification / floutage des visages → *étape suivante* (`src/blur_faces.py`).

## 6. Nettoyage (`src/cleaning.py`)

| Étape | Lignes supprimées |
|---|---|
| Messages de bots / modérateurs | 13 |
| Textes vides ou < 3 mots (posts image conservés) | 8 |
| Doublons (même texte **et** même image) | 9 |
| Plafond contenu sexuel (20 %) | 0 |
| **858 → 828 messages** | |

Opérations sur le texte : normalisation Unicode (NFKC), suppression des URLs, des
mentions et des marqueurs `[removed]`, `#mot → mot`, suppression du Markdown,
minuscules.

Choix justifiés :
- **Emojis convertis en texte** (`😂 → :face_with_tears_of_joy:`) plutôt que supprimés :
  ils portent l'ironie (cf. avertissement du TP2).
- **Colonne `id` conservée** en plus de `date, clean_text, media, url, source` :
  clé nécessaire pour l'annotation du TP2.
- **Contenu sexuel plafonné à 20 %** pour éviter qu'un modèle apprenne « toxique = sexuel ».
  Grâce aux filtres de collecte (pas de mots-clés sexuels, NSFW exclu), il ne représente
  que 3,6 % du corpus : le biais a été évité à la source.

## 7. Résultats (détail : `mini_report.md`)

- 828 messages, 11 subreddits, 80 images (9,7 %), 62 mots en moyenne (médiane 31).
- 6 153 mots distincts, 53 % d'hapax.
- Pertinence : 87 % (insult) … 3 % (neutral) des messages contiennent un mot du lexique
  du harcèlement → les mots-clés séparent bien les groupes.
- Limites : groupe *threat* surtout composé de victimes, classe *hate* faible (forte modération
  de Reddit), faux positifs dus à la racinisation de la recherche plein texte.

## 8. Traçabilité (MLflow)

Expérience `TP1_Collecte` (`mlflow ui`) :

| Run | Paramètres | Métriques | Artefacts |
|---|---|---|---|
| `scrape_reddit` | fenêtre, volume par requête, date de scraping | n_rows, n_images | `collection_config.json` (mots-clés, subreddits), `reddit_raw.csv` |
| `cleaning` | min_words, sexual_cap, seed, traitement emojis/hashtags | lignes supprimées par étape, part de contenu sexuel | `cyberbullying_raw.csv`, `cyberbullying_raw_clean.csv` |
| `analysis` | — | statistiques, diversité lexicale, pertinence | `mini_report.md`, `tp1_distribution.png` |

Plusieurs runs `scrape_reddit` existent : 639 lignes (sans r/4chan, r/greentext), puis 858
(avec). Leur comparaison dans l'interface MLflow documente l'évolution du corpus.
