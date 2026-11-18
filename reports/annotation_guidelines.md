# Guide d'annotation — TP2

## 1. Échantillon annoté
600 publications (`src/select_for_annotation.py`, seed 42) : les 200 mèmes Hateful Memes
+ 400 publications Reddit réparties à parts égales entre les 9 groupes de requête du TP1.

## 2. Classes (texte)

| Classe | Critère |
|---|---|
| `non_toxic` | Aucune attaque. Inclut les victimes qui **racontent ou citent** un harcèlement. |
| `insult_mockery` | Attaque l'apparence, l'intelligence ou le comportement d'une **personne** (roasts inclus). |
| `hate_discrimination` | Vise un **groupe** (origine, religion, genre, orientation, handicap…) ou une personne *pour* ce groupe. |
| `threat_intimidation` | L'auteur **exprime** une intention de nuire (« kill yourself », « mow them down »). |
| `irony_provocation` | Sarcasme, trolling, chambrage (*cope, seethe, clown*) sans attaque directe claire. |

**Règles de décision**
1. Plusieurs classes possibles → priorité **threat > hate > insult > irony > non_toxic**.
2. Discours rapporté ≠ attaque : « il m'a dit *je sais où tu habites* » → `non_toxic`.
3. Insulte vers une personne précise (joueur, interlocuteur) → `insult_mockery` ;
   chambrage général entre supporters → `irony_provocation`.
4. Légendes de mèmes jugées **seules** (sans l'image).
5. Cas incertain → colonne `ambiguous = True`.

## 3. Cas ambigus fréquents (justification)

| Cas | Décision | Raison |
|---|---|---|
| Légende neutre seule, haineuse avec l'image (« one of the most sold items… ») | `non_toxic`, ambigu | Le texte seul n'attaque personne ; la haine vient de l'image → justifie le modèle multimodal (TP3). |
| Humour noir sans cible (blague sur une tragédie) | `irony_provocation`, ambigu | Choquant mais pas dirigé contre une personne ou un groupe. |
| Critique politique virulente (partis, gouvernements) | `irony_provocation` ou `non_toxic` | Une opinion politique n'est pas une haine contre un groupe protégé. |
| Victime qui décrit des violences / menaces reçues | `non_toxic` | Règle 2 : l'auteur n'attaque personne. |
| Roast consenti (r/RoastMe) | `insult_mockery` | Le message reste une moquerie dirigée vers une personne réelle. |
| Généralisation sur un sexe (« women lie a lot more ») | `hate_discrimination` | Vise un groupe (genre). |

## 4. Images
Étiquettes reprises du dataset Hateful Memes (annotateurs formés de Meta, Kiela et al. 2020),
qui jugent le mème **dans son ensemble** (image + texte incrusté) :
`1 → hateful_discriminatory` (100 images), `0 → neutral` (100 images) — `src/image_labels.py`.
Une ré-annotation manuelle serait longue et moins fiable (la haine d'un mème est subtile : une
même image peut être anodine ou haineuse selon son texte). Limite : 2 des 4 étiquettes images
du TP sont utilisées (`insulting_humiliating` et `ambiguous_ironic` absentes).

## 5. Annotateurs et limites
- **Texte** : `A_llm_claude` — annotation réalisée par un modèle de langage (Claude, Anthropic)
  selon ce guide, pour réduire le temps d'annotation. Limite : un seul annotateur, possible biais
  du modèle ; un contrôle humain d'un sous-échantillon (accord de Cohen) est recommandé.
- **Images** : `meta_expert` (labels d'origine du dataset).
