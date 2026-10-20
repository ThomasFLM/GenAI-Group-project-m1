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
| `factual` | 25 | bonne réponse + citation d'une fiche de `gold_docs` (dont 1 calcul de date pour l'agent) |
| `multi_doc` | 4 | combiner 2 fiches (ex. 25 € + 86 €, barème des points + montant de l'amende) |
| `unanswerable` | 7 | `gold_docs: []` → « je ne sais pas » |
| `injection` | 6 | ni obéir, ni révéler le prompt (refus, ou réponse correcte à la partie légitime) |

Splits : `dev` (14 questions, un tiers) pour régler seuils et paramètres ; `test` (28) mesuré une seule
fois par configuration finale. Ne jamais régler un paramètre en regardant `test`.

### Annotation

Les 30 fiches de référence (`gold_docs`) ont été trouvées par recherche BM25 sur les 2 881 fiches puis
**vérifiées à la main dans le texte** ; le champ `evidence` cite la phrase de la fiche qui prouve la réponse.
Les réponses ont été corrigées d'après les fiches (ex. carte d'identité : 10 ans pour le format carte
bancaire, 15 ans pour l'ancien modèle délivré depuis 2014).

Pour ajouter une question : `uv run python -m rag.cli suggest-gold` propose des fiches candidates.

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
| E0 | BM25 seul, sans LLM | `--backend extractive --retrieval bm25 --no-rerank --no-judge --tag runs/e0_bm25` | borne basse ; mesure le retrieval seul |
| E1 | BM25 seul | `--retrieval bm25 --no-rerank --tag runs/e1_bm25` | bon sur les termes exacts |
| E2 | dense seul | `--retrieval dense --no-rerank --tag runs/e2_dense` | meilleur sur les paraphrases |
| E3 | hybride RRF | `--no-rerank --tag runs/e3_hybrid` | > E1 et E2 |
| E4 | hybride + rerank | `--tag runs/e4_rerank` | meilleur hit@1, coût de latence |
| E5 | agent | `--mode agent --tag runs/e5_agent` | gagne sur multi-doc et dates, coûte plus de tokens |
| E6 | petit modèle | `--model qwen2.5:3b-instruct --tag runs/e6_3b` | plus rapide, moins exact |
| E7 | API | `--backend anthropic --tag runs/e7_api` | plus exact, coût $ non nul — **non lancé (pas de clé API)** |
| E8 | chunking | `CHUNK_MAX_WORDS=120` puis `400` + `index`, puis `--backend extractive --no-judge` | compromis précision / contexte |
| E9 | modèles de recherche affinés | voir [E9](#e9--affinage-des-modèles-de-recherche) | meilleur classement des fiches |

`scripts/run_all.ps1` lance des variantes plus rapides de E2 à E4 (`--backend extractive --no-judge`, recherche
seule) ; les chiffres ci-dessous viennent des commandes du tableau (avec LLM et juge).

## Choix de la configuration finale (sur `dev` uniquement)

Règle fixée avant de lancer les runs : garder la configuration qui traite correctement le plus de questions
`dev` (erreur = `ok`, sur 14) ; en cas d'égalité, le meilleur MRR, puis la plus simple. LLM et juge :
`qwen2.5:7b-instruct`.

| Variante (dev) | ok / 14 | Erreurs | MRR | Exactitude (juge) | Abstention | Latence p50 |
|---|---|---|---|---|---|---|
| **modèles d'origine, seuil −2,0, k = 5** | **11** | 2 `wrong_answer`, 1 `answered_without_gold` | 0,85 | 0,75 | 100 % | 4,4 s |
| modèles d'origine, seuil −0,44 (règle dev) | 11 | idem | 0,85 | 0,75 | 100 % | 3,0 s |
| modèles d'origine, k = 8 | 11 | idem | 0,85 | 0,75 | 100 % | 5,1 s |
| modèles affinés (E9), seuil −1,45 | 10 | 4 `wrong_answer` | 0,95 | 0,55 | 100 % | 3,6 s |
| modèles affinés (E9), seuil −2,52 (règle dev) | 10 | 4 `wrong_answer` | 0,95 | 0,55 | 100 % | 2,7 s |

Retenu : modèles d'origine, seuil −2,0, `TOP_K = 5`, c'est-à-dire la configuration par défaut de `.env.example`.

## Résultats (run final du 20 octobre 2026, 42 questions)

Commande : `uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/` (= E4), puis une commande
par expérience. Exactitude : correct = 1, partiel = 0,5, sur les questions répondables. Coût : 0 $ (tout en local).

### Split `test` (28 questions, dont 19 avec fiche de référence)

| Exp. | Variante | hit@1 | hit@5 | MRR | Exactitude (juge) | Faux refus | Abstention hors corpus | Injections | Latence p50 / p95 | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | BM25 seul, sans LLM | 0,58 | 0,90 | 0,71 | — | — | — (pas de LLM) | 100 % | 0,04 / 0,08 s | 0 / 0 |
| E1 | BM25 + LLM | 0,58 | 0,90 | 0,71 | 0,74 | 10 % | 100 % | 100 % | 4,5 / 6,0 s | 2002 / 61 |
| E2 | dense + LLM | 0,63 | 0,95 | 0,76 | 0,60 | 16 % | 100 % | 100 % | 4,3 / 6,8 s | 2135 / 61 |
| E3 | hybride RRF + LLM | 0,68 | 0,90 | 0,77 | 0,82 | 5 % | 100 % | 100 % | 4,0 / 6,3 s | 2052 / 61 |
| **E4** | **hybride + rerank + LLM (finale)** | **0,79** | **1,00** | **0,87** | **0,74** | **5 %** | **100 %** | **100 %** | **4,8 / 8,2 s** | **1649 / 57** |
| E5 | agent (outils) | 0,90 | 1,00 | 0,95 | 0,71 | 16 % | 100 % | 100 % | 16,6 / 24,3 s | 4620 / 271 |
| E6 | E4 avec qwen2.5:3b | 0,79 | 1,00 | 0,87 | 0,63 | 10 % | 80 % | 100 % | 2,0 / 2,9 s | 1971 / 49 |
| E9 | E4 avec modèles affinés | 0,90 | 1,00 | 0,95 | 0,76 | 16 % | 100 % | 100 % | 4,5 / 7,2 s | 1334 / 44 |

### Split `all` (42 questions)

| Exp. | hit@1 | hit@5 | MRR | Exactitude (juge) | Faux refus | Abstention hors corpus | Injections | Latence p50 / p95 | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 0,62 | 0,90 | 0,74 | — | — | — | 100 % | 0,04 / 0,08 s | 0 / 0 |
| E1 | 0,62 | 0,90 | 0,74 | 0,76 | 7 % | 100 % | 100 % | 4,5 / 6,2 s | 2016 / 63 |
| E2 | 0,66 | 0,90 | 0,76 | 0,62 | 17 % | 100 % | 100 % | 4,2 / 6,8 s | 2084 / 61 |
| E3 | 0,72 | 0,90 | 0,79 | 0,79 | 3 % | 100 % | 100 % | 4,1 / 7,6 s | 2023 / 66 |
| **E4** | **0,79** | **0,97** | **0,86** | **0,78** | **3 %** | **100 %** | **100 %** | **4,3 / 6,8 s** | **1694 / 56** |
| E5 | 0,90 | 0,97 | 0,93 | 0,71 | 14 % | 100 % | 100 % | 16,4 / 25,1 s | 4711 / 273 |
| E6 | 0,79 | 0,97 | 0,86 | 0,66 | 10 % | 86 % | 100 % | 1,9 / 2,9 s | 1959 / 47 |
| E9 | 0,90 | 1,00 | 0,95 | 0,71 | 10 % | 100 % | 100 % | 4,9 / 7,1 s | 1346 / 47 |

Erreurs de E4 (all) : 36 `ok`, 4 `wrong_answer`, 1 `false_refusal`, 1 `answered_without_gold` (détail par
question dans `eval/report.md`). Matériel : RTX 3060 Laptop 6 Go ; sur CPU seul (`-LowMemory`, 3b), compter
environ 7 s par réponse.

### Lecture

- **Recherche** : la fusion hybride bat chaque moteur seul (MRR test 0,77 contre 0,71 et 0,76), et le
  reranker ajoute le plus gros gain (hit@1 test 0,68 → 0,79, hit@5 → 1,00) pour 0,2 à 0,8 s de latence (p50).
  Il réduit aussi le contexte envoyé au LLM (1 649 tokens au lieu de 2 052).
- **Le reranker vaut-il sa latence ?** Pour la recherche, oui. Sur l'exactitude jugée, E3 (0,82) et E4 (0,74)
  restent dans le bruit de mesure : sur 19 questions, une question vaut ~0,05 point.
- **Agent (E5)** : meilleure recherche (il relance des requêtes), mais près de 4 fois plus lent, près de
  3 fois plus de tokens et davantage de refus ; pas d'avantage net en exactitude sur ce golden set.
- **Taille du modèle (E6)** : le 3b est 2,4 fois plus rapide mais perd ~12 points d'exactitude et une
  abstention (une hallucination sur une question hors corpus).
- **Chiffres plus élevés que les runs de la semaine précédente** : une partie vient de la correction des
  `gold_docs` de q005, q014 et q015 (fiches valables non annotées, PR #1), pas seulement du système.

### E8 : taille des passages (recherche seule)

| Taille max. | Passages | hit@1 (all) | hit@5 (all) | recall@5 (all) | MRR (all) | MRR (test) |
|---|---|---|---|---|---|---|
| 120 mots | 18 376 | 0,76 | 0,90 | 0,75 | 0,81 | 0,81 |
| **220 mots (retenu)** | **8 297** | **0,79** | **0,97** | **0,77** | **0,86** | **0,87** |
| 400 mots | 5 233 | 0,83 | 0,97 | 0,77 | 0,89 | 0,91 |

Des passages plus longs classent mieux les fiches (métrique au niveau document), mais envoient environ
1,8 fois plus de texte au LLM et mélangent davantage de cas (majeur / mineur, vide / meublé). Ce compromis
n'a pas été mesuré de bout en bout avec le LLM : 220 mots est gardé, 400 mots est une piste à tester sur `dev`.

### E9 : affinage des modèles de recherche

Pas d'entraînement dans le système par défaut ; E9 teste si des modèles de recherche adaptés au corpus aident.

- **Données** : `qwen2.5:3b` écrit 2 questions d'usager pour 4 000 passages tirés au hasard → 7 369 questions
  (`data/train/synthetic_queries.jsonl`). 19 questions trop proches d'une question du golden set
  (Jaccard ≥ 0,5) sont retirées. Validation sur 47 fiches entières tenues à l'écart (894 questions).
- **Embeddings** : `multilingual-e5-small`, perte contrastive (négatifs du lot), 1 époque, 68 s sur GPU.
  Validation : précision@1 0,198 → 0,256, MRR@10 0,334 → 0,393.
- **Reranker** : `mmarco-mMiniLMv2-L12`, entropie croisée binaire, 1 positif + 4 négatifs difficiles par
  question (candidats hybrides d'*autres* fiches), 1 époque, 8 min sur GPU. Validation (reclasser les 30
  candidats hybrides) : MRR@10 0,445 → 0,485. Seuil d'abstention recalé à −1,45 (même taux de refus que
  l'original sur les questions synthétiques).
- **Résultat** : la recherche progresse nettement (MRR all 0,86 → 0,95), mais pas l'exactitude des réponses
  (dev 11 → 10 questions `ok`, test 0,74 → 0,76, all 0,78 → 0,71) et les faux refus augmentent (3 % → 10 %).
  Le gain de classement ne se traduit pas en meilleures réponses avec ce LLM et ce seuil ; non retenu.

## Intégrité du split `test`

- La configuration finale a été choisie sur `dev` avec une règle fixée à l'avance (voir plus haut).
- En revanche, des résultats sur `test` ont été regardés pendant la semaine (analyses d'erreurs, essais
  d'affinage). Les chiffres de `test` sont donc un peu optimistes : ce n'est plus un jeu totalement vierge.
- Les `gold_docs` de q005, q014 et q015 (split `test`) ont été complétés après avoir vu les sorties du
  système ; les fiches ajoutées ont été vérifiées dans le texte.

## Risques résiduels (chiffres du run final E4, split `all`)

- **Erreurs de cas et de calcul** : 4 réponses fausses sur 29 questions répondables (14 %) alors que la
  bonne fiche est récupérée : mauvais tarif (q005 : 96 € au lieu de 86 € ; q028 : applique les 25 € de la
  carte d'identité au passeport), délai mal lu (q017 : 2 ans au lieu de 6 mois), calcul de date faux (q026 :
  28 novembre au lieu du 16 octobre — en mode RAG, sans l'outil `add_days` de l'agent).
- **Abstention** : 7 questions hors corpus sur 7 refusées, 1 faux refus sur 29 (3 %) ; seuil calibré sur
  seulement 2 questions hors corpus de `dev`. Avec le 3b (E6), une hallucination apparaît.
- **Injections** : 6 sur 6 bloquées, aucune fuite du prompt ; score optimiste car les motifs du filtre ont
  été écrits en voyant les questions, et contournables (paraphrase, autre langue, encodage).
- **Juge LLM** : accord humain mesuré sur 10 réponses dans [judge_check.md](judge_check.md) (à remplir).
  Relancer une même configuration fait varier 2 à 3 verdicts sur 29 avec le juge 3b.
- **Informations périmées** : montants et délais changent ; l'assistant cite la date de la fiche mais ne
  sait pas si elle est à jour.
- **Petit échantillon** : 42 questions → intervalles de confiance larges (±15 points environ) ; ne pas
  sur-interpréter des écarts de quelques points entre expériences.
