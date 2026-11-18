# TP1 — Collecte et exploration des données
**Projet fil rouge MLOps : Cyberbullying Detection sur données multimodales (texte + image)**
FST — IDS 5 — 2026-2027

Constituer un corpus multimodal (texte + images) de messages potentiellement liés au
cyberharcèlement, nettoyé, documenté et traçable (MLflow + DVC), qui servira de base au TP2.

---

## 1. Structure du projet

```
TP_FR/
├── data/                               # versionné par DVC (pas par Git)
│   ├── reddit_raw.csv                  # sortie brute Reddit
│   ├── hateful_memes_raw.csv           # sortie brute Hateful Memes
│   ├── cyberbullying_raw.csv           # corpus brut fusionné (toutes les métadonnées)
│   ├── cyberbullying_raw_clean.csv     # corpus nettoyé
│   ├── images/                         # 200 mèmes (hm_<id>.jpg)
│   └── *.dvc                           # empreintes DVC (versionnées par Git)
├── notebooks/
│   └── TP1_collecte.ipynb              # Partie 4 : analyse + mini-rapport
├── src/
│   ├── reddit_scraper.py               # source 1 : Reddit (texte)
│   ├── hateful_memes_sampler.py        # source 2 : Hateful Memes (texte + image)
│   ├── cleaning.py                     # Partie 3 : fusion + nettoyage
│   └── fourchan_scraper.py             # non utilisé (robots.txt, voir §3)
├── reports/tp1_distribution.png
├── mini_report.md                      # Partie 4.3 (généré par le notebook)
├── mlflow.db / mlruns/                 # Partie 6 : suivi MLflow
├── .dvc/                               # configuration DVC
├── requirements.txt
└── README.md
```

`src/` et `reports/` s'ajoutent à la structure demandée : la collecte est faite par des
scripts réutilisables, le notebook se limite à l'analyse.

## 2. Reproduire

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Option A : récupérer les données versionnées
dvc pull

# Option B : tout régénérer
python src/reddit_scraper.py          # ~15 min -> data/reddit_raw.csv
python src/hateful_memes_sampler.py   # ~3 min  -> data/hateful_memes_raw.csv + images
python src/cleaning.py                # -> cyberbullying_raw.csv + cyberbullying_raw_clean.csv
jupyter notebook notebooks/TP1_collecte.ipynb   # Run All -> mini_report.md
mlflow ui                             # expérience « TP1_Collecte »
```

Fenêtre temporelle fixe et `seed=42` partout : une nouvelle exécution redonne le même
corpus (aux suppressions Reddit près).

## 3. Choix des sources

| Source | Décision | Raison |
|---|---|---|
| **Reddit** (archive Arctic Shift) | ✅ source 1 — texte | Gratuit, sans clé, public, score de votes. L'API officielle (PRAW) n'accepte plus de nouvelles clés depuis le 31/10/2026 ; Arctic Shift est le successeur de Pushshift. |
| **Hateful Memes** (Meta, Kiela et al. 2020 — miroir Hugging Face `cs5242-hateful-memes/hateful-memes-data`) | ✅ source 2 — texte + image | Mèmes annotés haineux / non haineux : vraies paires texte-image, sans visage de personne privée. Licence Meta *non-commercial research* : citée ici, images non republiées. |
| X / Twitter | ❌ | Plus de lecture gratuite (API payante), snscrape et Nitter hors service. |
| 4chan (API officielle) | ❌ | `robots.txt` de `a.4cdn.org` : `Disallow: /` → respecté. `fourchan_scraper.py` vérifie le robots.txt et s'arrête ; il est conservé comme preuve. Le contenu 4chan reste présent indirectement via r/4chan et r/greentext. |
| YouTube | ❌ (choix) | Commentaires jugés peu représentatifs du harcèlement. |

**Pourquoi pas d'images Reddit ?** La v1 contenait 80 mèmes Reddit (r/memes, r/dankmemes,
r/4chan, r/greentext) : contenu aléatoire, sans lien avec le harcèlement. Les images
réellement utiles (photos humiliantes de personnes réelles, captures nominatives) sont
interdites par la protection de la vie privée. Elles ont donc été remplacées par un
échantillon du dataset Hateful Memes (v2).

## 4. Périmètre de collecte

### Source 1 — Reddit
- **Fenêtre** : 01/05/2026 → 01/11/2026 (messages de plus de 36 h : scores définitifs).
- **Volume** : 60 commentaires max par couple (subreddit, groupe) ; 200 pour RoastMe.

| Groupe de requête | Mots-clés (recherche plein texte) | Subreddits |
|---|---|---|
| insult | ugly, loser, stupid, idiot, pathetic, fat | RoastMe, gaming, 4chan |
| roast | *(aucun)* — commentaires de 1er niveau, score ≥ 5 | RoastMe |
| threat | "kill yourself", kys, "watch your back", "find you", "hurt you" | bullying, TrueOffMyChest, 4chan |
| threat_quoted | "kill you", "know where you live", "threatened to", "you're dead"… | legaladvice, relationship_advice |
| hate | subhuman, "go back to", disgusting, vermin, worthless | TrueOffMyChest, unpopularopinion, 4chan |
| hate_immigration | invaders, illegals, parasites, "deport them", "third world"… | europe, unitedkingdom, immigration |
| irony | cope, seethe, ratio, "nobody asked", "ok boomer" | memes, unpopularopinion, greentext |
| provocation_sports | cope, seethe, clown, bottlers, "cry more", "stay mad" | soccer, nba |
| neutral | *(aucun — échantillon aléatoire)* | AskReddit, CasualConversation |

Le `query_group` indique **comment** un message a été trouvé, **pas** son étiquette
(l'annotation est faite au TP2). Les subreddits du groupe `hate_immigration` sont
généralistes (non liés à un camp politique) pour que le modèle apprenne la haine, pas
une opinion.

### Source 2 — Hateful Memes
- Split `train`, lu en **streaming** (le dataset complet fait ~8 Go : rien n'est
  téléchargé au-delà du nécessaire), mélangé (`seed=42`, buffer 2000).
- **Échantillon équilibré** : 100 mèmes haineux (label 1) + 100 non haineux (label 0).
  Le label d'origine est conservé dans `metadata` (contrôle de l'annotation au TP2).
- Images redimensionnées à 512 px max, enregistrées en JPEG.

## 5. Légalité et éthique

- Données publiques uniquement, via des API documentées ; ~1 requête/s (gestion du code 429).
- `robots.txt` vérifié (4chan exclu pour cette raison).
- **Aucune donnée personnelle** : pseudonymes remplacés par un hash SHA-256 salé
  (`author_hash`), mentions `u/…` et `@…` supprimées du texte.
- NSFW exclu à la collecte (`over_18=false`) ; aucune image de personne privée.
- Hateful Memes : licence de recherche non commerciale respectée (citation, pas de
  republication ; le stockage DVC est local).

## 6. Nettoyage (`src/cleaning.py`)

| Étape | Lignes supprimées |
|---|---|
| Fusion Reddit (1 346) + Hateful Memes (200) | = 1 546 |
| Messages de bots / modérateurs | 13 |
| Textes vides ou < 3 mots (lignes avec image conservées) | 13 |
| Doublons (même texte **et** même image) | 10 |
| Plafond contenu sexuel (20 %) | 0 |
| **Total : 1 546 → 1 510 messages** | |

Opérations sur le texte : normalisation Unicode (NFKC), suppression des URLs, des
mentions et des marqueurs `[removed]`, `#mot → mot`, suppression du Markdown, minuscules.

Choix justifiés :
- **Emojis convertis en texte** (`😂 → :face_with_tears_of_joy:`) plutôt que supprimés :
  ils portent l'ironie (cf. avertissement du TP2).
- **Colonne `id` conservée** en plus de `date, clean_text, media, url, source` : clé
  nécessaire pour l'annotation du TP2.
- **Contenu sexuel plafonné à 20 %** pour éviter qu'un modèle apprenne « toxique = sexuel ».
  Grâce aux filtres de collecte (pas de mots-clés sexuels, NSFW exclu), il ne représente
  que 3,8 % du corpus : le biais a été évité à la source.
- Doublons définis sur (texte, image) : Hateful Memes contient volontairement des mèmes
  à légende identique mais image différente (« confondeurs »), qui changent de sens.

## 7. Résultats (détail : `mini_report.md`)

- **1 510 messages** : Reddit 1 310 (17 subreddits) + Hateful Memes 200 ; 200 avec image (13,2 %).
- Longueur moyenne 54 mots (médiane 25) : distribution très asymétrique.
- 8 492 mots distincts, 51 % d'hapax : vocabulaire riche.
- Pertinence (part des messages contenant un mot du lexique du harcèlement) :
  insult 87 %, threat_quoted 74 %, irony 56 %, hate 43 %, provocation 41 %,
  threat 36 %, hate_immigration 34 %, neutral 5 %, **roast 2 %**.
- Constat clé : les roasts se moquent sans mots « toxiques » classiques → un lexique ne
  suffit pas, il faut un modèle qui comprend le contexte (TP3).
- Limites : *threat* = surtout des victimes qui racontent/citent des menaces ; RoastMe =
  moquerie consentie ; faux positifs dus à la racinisation de la recherche plein texte ;
  images non liées aux textes Reddit.

## 8. Traçabilité

### MLflow — expérience `TP1_Collecte`

| Run | Paramètres | Métriques | Artefacts |
|---|---|---|---|
| `scrape_reddit` | fenêtre, volume par requête, date de scraping | n_rows, n_images | `collection_config.json` (mots-clés, subreddits, réglages par groupe), `reddit_raw.csv` |
| `sample_hateful_memes` | dataset, split, n_per_label, seed, licence | n_memes, n_hateful, rows_streamed | `hateful_memes_raw.csv` |
| `cleaning` | min_words, sexual_cap, seed, traitement emojis/hashtags | lignes supprimées par étape, part de contenu sexuel | `cyberbullying_raw.csv`, `cyberbullying_raw_clean.csv` |
| `analysis` | — | statistiques, diversité lexicale, pertinence | `mini_report.md`, `tp1_distribution.png` |

### DVC — versions des données

| Tag Git | Contenu |
|---|---|
| `data-v1` | Reddit seul : 828 messages, 80 mèmes Reddit (aléatoires) |
| `data-v2` | Reddit (+ RoastMe, hate_immigration, provocation_sports, threat_quoted) + Hateful Memes : 1 510 messages, 200 mèmes |

```bash
dvc diff data-v1 data-v2                  # fichiers modifiés entre deux versions
git checkout data-v1 && dvc checkout      # revenir aux données v1
git checkout master  && dvc checkout      # revenir à la dernière version
```

Remote DVC : dossier local `../dvc_storage` (données non publiées).
