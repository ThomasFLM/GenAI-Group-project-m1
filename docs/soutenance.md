# Soutenance — 21 octobre 2026 (10 min + 5 min de questions)

Chacun présente une partie (≈ 2 min 30 chacun pour 4 personnes).

| # | Partie | Durée | Contenu | Supports |
|---|---|---|---|---|
| 1 | Données | 2 min 30 | Problème (assistant démarches), choix du corpus Service-Public, sélection des ~500 fiches, nettoyage, statistiques, ce qu'on a vu en lisant 20 fiches | data card, `corpus_stats.json` |
| 2 | Architecture | 2 min 30 | Découpage par section + en-tête, hybride BM25 + dense + RRF, reranker, sortie JSON validée, agent et outils, garde-fous | schéma du README, architecture.md |
| 3 | Démo | 2 min 30 | 1 question simple avec citation · 1 question date (agent + `add_days`) · 1 question hors corpus (« je ne sais pas ») · 1 injection bloquée — montrer la trace (tokens, latence, outils) | Streamlit **+ vidéo enregistrée en secours** |
| 4 | Évaluation | 2 min 30 | Golden set, tableau des expériences (E0→E7), compromis qualité / coût / latence, analyse d'erreurs (2 exemples concrets corrigés), risques résiduels | report.md, evaluation.md |

## Questions probables du jury

- Pourquoi ces tailles de passages ? → expérience E8 (chiffres).
- Le reranker vaut-il sa latence ? → E3 vs E4 (hit@1, p95).
- Comment savez-vous que le juge est fiable ? → accord avec annotation humaine sur 10 réponses.
- Que se passe-t-il si un montant change ? → risque « périmé », date affichée, re-téléchargement.
- Votre résistance aux injections est-elle surestimée ? → oui, motifs écrits en voyant les questions ; injections inédites testées sur `test`.

## Check-list avant le 20 octobre 23:59

- [ ] `gold_docs` annotés et réponses de référence vérifiées dans les fiches
- [ ] runs E0–E7 lancés, tableau de résultats rempli
- [ ] data card complétée (chiffres + observations)
- [ ] versions exactes des modèles dans le README (`ollama list`)
- [ ] vidéo de démo enregistrée
- [ ] `git tag v1.0 && git push origin v1.0`
