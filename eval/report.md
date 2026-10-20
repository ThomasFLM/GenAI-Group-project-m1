# Evaluation report

Model: `qwen2.5:7b-instruct` · mode: `rag` · retrieval: `hybrid` · embeddings: `intfloat/multilingual-e5-small` · reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

## Split: all (42 questions)

| Block | Metric | Value |
|---|---|---|
| retrieval | n_labeled | 29 |
| retrieval | hit@1 | 0.793 |
| retrieval | hit@5 | 0.966 |
| retrieval | recall@5 | 0.767 |
| retrieval | mrr | 0.862 |
| answers | judge_accuracy | 0.776 |
| answers | token_recall | 0.716 |
| answers | number_recall | 0.784 |
| answers | answer_rate | 0.966 |
| answers | false_refusal_rate | 0.034 |
| answers | citation_precision | 0.821 |
| answers | citation_recall | 0.723 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 3.743 |
| cost_latency | latency_p50_s | 4.285 |
| cost_latency | latency_p95_s | 6.795 |
| cost_latency | tokens_in_mean | 1693.714 |
| cost_latency | tokens_out_mean | 56.5 |
| cost_latency | tokens_in_total | 71136 |
| cost_latency | tokens_out_total | 2373 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.81 |

Stop reasons: `{'stop': 28, 'validation_failed': 3, 'low_retrieval_score': 5, 'injection_blocked': 6}`

Errors: `{'ok': 36, 'answered_without_gold': 1, 'wrong_answer': 4, 'false_refusal': 1}`

## Split: dev (14 questions)

| Block | Metric | Value |
|---|---|---|
| retrieval | n_labeled | 10 |
| retrieval | hit@1 | 0.8 |
| retrieval | hit@5 | 0.9 |
| retrieval | recall@5 | 0.825 |
| retrieval | mrr | 0.85 |
| answers | judge_accuracy | 0.85 |
| answers | token_recall | 0.817 |
| answers | number_recall | 0.833 |
| answers | answer_rate | 1.0 |
| answers | false_refusal_rate | 0.0 |
| answers | citation_precision | 0.85 |
| answers | citation_recall | 0.675 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 3.055 |
| cost_latency | latency_p50_s | 3.323 |
| cost_latency | latency_p95_s | 4.824 |
| cost_latency | tokens_in_mean | 1783.429 |
| cost_latency | tokens_out_mean | 55.143 |
| cost_latency | tokens_in_total | 24968 |
| cost_latency | tokens_out_total | 772 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.857 |

Stop reasons: `{'stop': 10, 'validation_failed': 1, 'low_retrieval_score': 1, 'injection_blocked': 2}`

Errors: `{'ok': 12, 'answered_without_gold': 1, 'wrong_answer': 1}`

## Split: test (28 questions)

| Block | Metric | Value |
|---|---|---|
| retrieval | n_labeled | 19 |
| retrieval | hit@1 | 0.789 |
| retrieval | hit@5 | 1.0 |
| retrieval | recall@5 | 0.737 |
| retrieval | mrr | 0.868 |
| answers | judge_accuracy | 0.737 |
| answers | token_recall | 0.664 |
| answers | number_recall | 0.755 |
| answers | answer_rate | 0.947 |
| answers | false_refusal_rate | 0.053 |
| answers | citation_precision | 0.806 |
| answers | citation_recall | 0.75 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 4.087 |
| cost_latency | latency_p50_s | 4.777 |
| cost_latency | latency_p95_s | 8.213 |
| cost_latency | tokens_in_mean | 1648.857 |
| cost_latency | tokens_out_mean | 57.179 |
| cost_latency | tokens_in_total | 46168 |
| cost_latency | tokens_out_total | 1601 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.786 |

Stop reasons: `{'stop': 18, 'validation_failed': 2, 'low_retrieval_score': 4, 'injection_blocked': 4}`

Errors: `{'ok': 24, 'wrong_answer': 3, 'false_refusal': 1}`

## Error analysis (failed questions)

| id | category | error | question | expected | got | retrieved |
|---|---|---|---|---|---|---|
| q004 | factual | **answered_without_gold** | Combien de temps un passeport est-il valable pour un adulte ? | 10 ans | Un passeport valide 1 an peut être délivré en urgence à un adulte pour des motifs d'ordre médical, humanitaires ou professionnels. | F1373, F21092, F1035, F21244, F21245 |
| q005 | factual | **wrong_answer** | Quel est le prix du timbre fiscal pour un passeport d'adulte ? | 86 € (en métropole) | Le prix du timbre fiscal pour un passeport d'adulte est de 96 €. | F14929, F1344, F21245, F14930, F21091 |
| q017 | factual | **wrong_answer** | Combien de temps avant la fin du bail un propriétaire doit-il donner congé pour un logement vide ? | Au moins 6 mois avant la date d'échéance du bail | Le propriétaire doit donner congé pour un logement vide au plus tard 2 ans avant la date d'échéance du bail en cours. | F929, F1168 |
| q026 | factual | **wrong_answer** | J'ai reçu un avis de contravention envoyé le 1er septembre 2026. Jusqu'à quelle date puis-je le contester ? | Jusqu'au 16 octobre 2026 (45 jours après l'envoi de l'avis) | Vous avez jusqu'au 28 novembre 2026 pour contester l'avis de contravention, soit 45 jours à compter de la date d'envoi de l'avis le 1er sept | F920, F17698, F34829, F18509, F34212 |
| q028 | multi_doc | **wrong_answer** | J'ai perdu ma carte d'identité et mon passeport : combien vais-je payer en timbres fiscaux pour refaire les deux ? | 111 € : 25 € pour la carte d'identité + 86 € pour le passeport | Vous devrez payer 50 € en timbres fiscaux pour renouveler à la fois votre carte d'identité et votre passeport, car chaque document coûte 25  | F33805, F1344, F21244 |
| q029 | multi_doc | **false_refusal** | Pour une location meublée, quel est le dépôt de garantie maximal et quelle est la durée du bail ? | Dépôt de garantie de 2 mois de loyer hors charges maximum ; bail d'1 an minimum (9 mois pour un étudiant) | Je ne sais pas : cette information ne figure pas dans les documents dont je dispose. | F31269 |

Legend: `retrieval_miss` = the right document is not retrieved; `false_refusal` = retrieved but refused; `wrong_answer` = retrieved but wrong answer; `wrong_citation` = right answer, wrong source; `hallucination` = answers although the information is not in the corpus; `injection_followed`/`prompt_leak` = successful injection; `unlabeled` = gold_docs to annotate.
