# Mini-rapport TP1 — Collecte
1. **Sources** : (a) Reddit via l'archive Arctic Shift (texte) ; (b) Hateful Memes (Meta, Kiela et al. 2020), échantillon équilibré haineux / non haineux lu en streaming (texte + image). X exclu (API payante), 4chan exclu (robots.txt `Disallow: /`, présent via r/4chan et r/greentext).
2. **Subreddits** : 17 (4chan, AskReddit, CasualConversation, RoastMe, TrueOffMyChest, bullying, europe, gaming, greentext, immigration, legaladvice, memes, nba, relationship_advice, soccer, unitedkingdom, unpopularopinion) ; fenêtre 01/05–01/11/2026.
3. **Mots-clés** : insult (ugly, loser, idiot…), threat (kys, watch your back…), hate (subhuman, invaders, illegals…), irony/provocation (cope, seethe, clown…) + RoastMe (roasts votés ≥ 5) + échantillon neutre.
4. **Taille** : 1546 messages bruts → 1510 après nettoyage (reddit : 1310, hateful_memes : 200), dont 200 avec image (13.2%) ; longueur moyenne 54 mots.
5. **Diversité** : 8,492 mots distincts, TTR 0.109, 51% d'hapax → vocabulaire riche et varié.
6. **Pertinence** : 37% des messages contiennent un mot du lexique du harcèlement (vs 5% pour le groupe neutre).
7. **Bruit** : messages de bots/modérateurs, marqueurs [removed] et doublons supprimés ; la recherche plein-texte (stemming) ramène des faux positifs (« find you » → « finding my purpose »).
8. **Biais** : le groupe *threat* contient surtout des victimes racontant le harcèlement, et la classe *hate* est sous-représentée (Reddit modère fortement).
9. **Contenu sexuel** : plafonné à 20 % ; seulement 3.8% après filtrage NSFW à la collecte.
10. **Limite** : les images (mèmes) ne sont pas liées aux textes Reddit ; les images Reddit de la v1 (mèmes aléatoires) ont été abandonnées faute de pertinence.
