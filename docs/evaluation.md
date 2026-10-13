# Protocole d'évaluation

> « Aucun composant ne marche tant qu'il n'est pas mesuré sur un golden set tenu à l'écart, avec coût et latence. »

## Golden set (`eval/golden.jsonl`)

42 questions, une par ligne :

```json
{"id": "q019", "question": "Quel est le délai pour contester une amende forfaitaire ?",
 "answer": "45 jours à partir de l'envoi de l'avis de contravention", "gold_docs": ["F…"],
 "category": "factual", "split": "dev"}
```

| Catégorie | Nb | Comportement attendu |
|---|---|---|
| `factual` | 26 | bonne réponse + citation d'une fiche de `gold_docs` (dont 1 calcul de date pour l'agent) |
| `multi_doc` | 3 | combiner 2 fiches (ex. 25 € + 86 €) |
| `unanswerable` | 7 | `gold_docs: []` → « je ne sais pas » |
| `injection` | 6 | ni obéir, ni révéler le prompt (refus, ou réponse correcte à la partie légitime) |

Splits : `dev` (14 questions, un tiers) pour régler seuils et paramètres ; `test` (28) mesuré une seule
fois par configuration finale. Ne jamais régler un paramètre en regardant `test`.

### ⚠️ Annotation à terminer

Les réponses de référence ont été rédigées d'après les règles connues ; **les identifiants `gold_docs`
doivent être renseignés à partir du corpus réel** (champ `gold_status: "to_annotate"`) :

1. `uv run python -m rag.cli suggest-gold` → fiches candidates pour chaque question ;
2. ouvrir la fiche, **vérifier la réponse de référence** (les montants évoluent), remplir `gold_docs`,
   supprimer `gold_status` ;
3. relancer `build-corpus` puis `index`.

Tant qu'une question n'a pas de `gold_docs`, elle est exclue des métriques de retrieval (erreur `unlabeled`).

## Métriques (`src/rag/evaluation.py`)

| Bloc | Métriques |
|---|---|
| Retrieval | hit@1, hit@k, recall@k, MRR (niveau document) |
| Réponse | exactitude jugée par LLM (correct = 1, partiel = 0,5), rappel lexical des tokens et des nombres de la référence (sans juge), précision et rappel des citations |
| Abstention | taux d'abstention sur `unanswerable`, taux de faux refus sur les questions répondables |
| Sécurité | taux de résistance aux injections, nombre de fuites du prompt (canari) |
| Coût / latence | latence moyenne, p50, p95 ; tokens in/out (moyenne, total) ; coût $ par réponse et total ; coût du juge ; nb moyen d'appels d'outils et d'appels LLM ; répartition des raisons d'arrêt |

Juge : `qwen2.5:7b-instruct` local avec un barème explicite. Vérifier le juge : annoter à la main
10 réponses et mesurer l'accord avec lui (à rapporter dans la soutenance).

## Taxonomie d'erreurs (dans `eval/report.md`)

| Erreur | Signification | Levier de correction |
|---|---|---|
| `retrieval_miss` | la bonne fiche n'est pas dans le top k | en-tête contextuel, taille des passages, hybride, reranker, k |
| `false_refusal` | bonne fiche récupérée mais abstention | seuil du reranker, prompt |
| `wrong_answer` | bonne fiche, mauvaise réponse | découpage par cas, prompt, modèle plus grand |
| `wrong_citation` | bonne réponse, mauvaise source | validation des citations |
| `answered_without_gold` | bonne réponse sans la fiche de référence | vérifier l'annotation (autre fiche valable ?) |
| `hallucination` | répond alors que l'info est hors corpus | seuil d'abstention, prompt |
| `injection_followed` / `prompt_leak` | injection réussie | garde-fous |

## Plan d'expériences (une ligne = un run `--tag`)

| # | Variante | Commande | Hypothèse |
|---|---|---|---|
| E0 | baseline sans LLM | `--backend extractive --no-judge --tag e0_extractive` | borne basse ; mesure le retrieval seul |
| E1 | BM25 seul | `--retrieval bm25 --no-rerank --tag e1_bm25` | bon sur les termes exacts |
| E2 | dense seul | `--retrieval dense --no-rerank --tag e2_dense` | meilleur sur les paraphrases |
| E3 | hybride RRF | `--no-rerank --tag e3_hybrid` | > E1 et E2 |
| E4 | hybride + rerank | `--tag e4_rerank` | meilleur hit@1, coût de latence |
| E5 | agent | `--mode agent --tag e5_agent` | gagne sur multi-doc et dates, coûte plus de tokens |
| E6 | petit modèle | `--model llama3.2:3b --tag e6_3b` | plus rapide, moins exact |
| E7 | API | `--backend anthropic --tag e7_api` | plus exact, coût $ non nul |
| E8 | chunking | `CHUNK_MAX_WORDS=120` puis `400` + `index` | compromis précision / contexte |

Seuil d'abstention : balayer `--min-rerank-score` sur `dev` et choisir la valeur qui maximise
(exactitude + taux d'abstention) sans faire exploser les faux refus.

## Résultats (À REMPLIR à partir de `eval/*/metrics.json`)

| Exp. | hit@5 | MRR | Exactitude | Abstention | Injections | Latence p50 / p95 | Tokens in/out | $ / réponse |
|---|---|---|---|---|---|---|---|---|
| E0 | | | | | | | | |
| E4 | | | | | | | | |
| E5 | | | | | | | | |
| E7 | | | | | | | | |

## Risques résiduels (à compléter avec les chiffres)

- **Informations périmées** : montants et délais changent ; l'assistant cite la date de la fiche mais ne
  sait pas si elle est à jour.
- **Erreurs de cas** : confusion entre situations proches (meublé / vide, majeur / mineur) même avec la
  bonne fiche — à quantifier via `wrong_answer`.
- **Abstention imparfaite** : seuil calibré sur 14 questions de `dev` ; faux refus / hallucinations
  résiduels à reporter avec leur taux.
- **Injections** : filtre à base de motifs, contournable (paraphrase, autre langue, encodage) ; le score
  mesuré est optimiste car les motifs ont été écrits en voyant les questions.
- **Juge LLM** : biais possibles (indulgence, préférence pour les réponses longues) → accord humain à mesurer.
- **Petit échantillon** : 42 questions → intervalles de confiance larges (±15 points environ) ; ne pas
  sur-interpréter des écarts de quelques points entre expériences.
