# Mini-rapport TP1 — Collecte
1. **Source** : Reddit via l'archive Arctic Shift (API Reddit fermée aux nouvelles clés depuis le 31/10/2026). X exclu (API payante, snscrape cassé) ; 4chan exclu (robots.txt : `Disallow: /`) mais présent indirectement via r/4chan et r/greentext.
2. **Subreddits** : 11 (4chan, AskReddit, CasualConversation, RoastMe, TrueOffMyChest, bullying, dankmemes, gaming, greentext, memes, unpopularopinion) ; fenêtre 01/05–01/11/2026.
3. **Mots-clés** : insult (ugly, loser, stupid, idiot…), threat (kill yourself, kys…), hate (subhuman, go back to…), irony (cope, seethe, ratio…) + échantillon neutre sans mot-clé.
4. **Taille** : 858 messages bruts → 828 après nettoyage, dont 80 avec image (9.7%) ; longueur moyenne 62 mots.
5. **Diversité** : 6,153 mots distincts, TTR 0.128, 53% d'hapax → vocabulaire riche et varié.
6. **Pertinence** : 43% des messages contiennent un mot du lexique du harcèlement (vs 3% pour le groupe neutre).
7. **Bruit** : messages de bots/modérateurs, marqueurs [removed] et doublons supprimés ; la recherche plein-texte (stemming) ramène des faux positifs (« find you » → « finding my purpose »).
8. **Biais** : le groupe *threat* contient surtout des victimes racontant le harcèlement, et la classe *hate* est sous-représentée (Reddit modère fortement).
9. **Contenu sexuel** : plafonné à 20 % ; seulement 3.6% après filtrage NSFW à la collecte.
10. **Limite** : une seule source (Reddit) — choix à valider avec l'enseignante.
