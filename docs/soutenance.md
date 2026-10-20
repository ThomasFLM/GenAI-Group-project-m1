# Soutenance — 21 octobre 2026 (10 min + 5 min de questions)

Chacun présente une partie (≈ 2 min 30 chacun pour 4 personnes).

| # | Partie | Durée | Contenu | Supports |
|---|---|---|---|---|
| 1 | Données | 2 min 30 | Problème (assistant démarches), choix du corpus Service-Public, sélection des ~500 fiches, nettoyage, statistiques, ce qu'on a vu en lisant 20 fiches | data card, `corpus_stats.json` |
| 2 | Architecture | 2 min 30 | Découpage par section + en-tête, hybride BM25 + dense + RRF, reranker, sortie JSON validée, agent et outils, garde-fous | schéma du README, architecture.md |
| 3 | Démo | 2 min 30 | 1 question simple avec citation · 1 question date (agent + `add_days`) · 1 question hors corpus (« je ne sais pas ») · 1 injection bloquée — montrer la trace (tokens, latence, outils) | Streamlit **+ vidéo enregistrée en secours** |
| 4 | Évaluation | 2 min 30 | Golden set, tableau des expériences (E0→E9), compromis qualité / coût / latence, analyse d'erreurs (2 exemples concrets), risques résiduels, intégrité du split `test` | report.md, evaluation.md |

Chiffres clés (E4, configuration finale, split `test`) : hit@5 1,00 · MRR 0,87 · exactitude jugée 0,74 ·
abstention hors corpus 100 % · injections bloquées 100 % · p50 4,8 s · ~1 650 tokens en entrée · 0 $.

## Questions probables du jury

- Pourquoi ces tailles de passages ? → E8 : 120 mots = MRR 0,81, 220 = 0,86, 400 = 0,89 en recherche seule ;
  400 mots envoie 1,8 fois plus de texte au LLM et mélange plus de cas, non testé de bout en bout → on garde 220.
- Le reranker vaut-il sa latence ? → E3 vs E4 : hit@1 test 0,68 → 0,79, hit@5 0,90 → 1,00, pour 0,2 à 0,8 s
  de plus (p50) et un contexte plus court (2 052 → 1 649 tokens).
- Pourquoi pas l'agent par défaut ? → E5 : meilleure recherche (MRR 0,95) mais près de 4 fois plus lent,
  près de 3 fois plus de tokens, et pas plus exact sur ce golden set.
- Avez-vous entraîné quelque chose ? → E9 : affinage des embeddings et du reranker sur 7 369 questions
  synthétiques ; meilleur classement (MRR 0,86 → 0,95) mais pas de meilleures réponses → non retenu.
- Comment savez-vous que le juge est fiable ? → accord avec annotation humaine sur 10 réponses
  ([judge_check.md](judge_check.md)) ; et le juge est passé du 3b au 7b, car le 3b variait de 2-3 verdicts d'un run à l'autre.
- Que se passe-t-il si un montant change ? → risque « périmé », date affichée, re-téléchargement.
- Votre résistance aux injections est-elle surestimée ? → oui, motifs écrits en voyant les questions.
- Avez-vous réglé des choses sur `test` ? → configuration finale choisie sur `dev` avec une règle fixée à
  l'avance ; mais `test` a été regardé pendant la semaine et 3 `gold_docs` corrigés → chiffres un peu optimistes.

## Check-list avant le 20 octobre 23:59

- [x] `gold_docs` annotés et réponses de référence vérifiées dans les fiches (q005, q014, q015 complétés, PR #1)
- [x] runs E0–E6, E8, E9 lancés, tableau de résultats rempli (E7 non lancé : pas de clé API)
- [ ] accord humain du juge sur 10 réponses ([judge_check.md](judge_check.md)) puis reporter le chiffre
- [x] data card complétée (chiffres + observations)
- [x] versions exactes des modèles dans le README
- [ ] vidéo de démo enregistrée
- [ ] fusion dans `main`, puis `git tag v1.0 && git push origin v1.0`
