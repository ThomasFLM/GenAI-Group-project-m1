# Rapport d'évaluation

Modèle : `qwen2.5:3b-instruct` · mode : `rag` · retrieval : `hybrid` · embeddings : `intfloat/multilingual-e5-small` · reranker : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

## Split : all (42 questions)

| Bloc | Métrique | Valeur |
|---|---|---|
| retrieval | n_labeled | 29 |
| retrieval | hit@1 | 0.69 |
| retrieval | hit@5 | 0.862 |
| retrieval | recall@5 | 0.71 |
| retrieval | mrr | 0.759 |
| answers | judge_accuracy | 0.759 |
| answers | token_recall | 0.596 |
| answers | number_recall | 0.691 |
| answers | answer_rate | 0.897 |
| answers | false_refusal_rate | 0.103 |
| answers | citation_precision | 0.635 |
| answers | citation_recall | 0.59 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 6.58 |
| cost_latency | latency_p50_s | 7.273 |
| cost_latency | latency_p95_s | 9.814 |
| cost_latency | tokens_in_mean | 1829.738 |
| cost_latency | tokens_out_mean | 46.024 |
| cost_latency | tokens_in_total | 76849 |
| cost_latency | tokens_out_total | 1933 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.881 |

Raisons d'arrêt : `{'stop': 26, 'abstained': 3, 'validation_failed': 2, 'low_retrieval_score': 5, 'injection_blocked': 6}`

Erreurs : `{'ok': 30, 'retrieval_miss': 2, 'answered_without_gold': 2, 'wrong_answer': 4, 'false_refusal': 3, 'wrong_citation': 1}`

## Split : dev (14 questions)

| Bloc | Métrique | Valeur |
|---|---|---|
| retrieval | n_labeled | 10 |
| retrieval | hit@1 | 0.8 |
| retrieval | hit@5 | 0.9 |
| retrieval | recall@5 | 0.825 |
| retrieval | mrr | 0.85 |
| answers | judge_accuracy | 0.8 |
| answers | token_recall | 0.679 |
| answers | number_recall | 0.683 |
| answers | answer_rate | 0.9 |
| answers | false_refusal_rate | 0.1 |
| answers | citation_precision | 0.778 |
| answers | citation_recall | 0.667 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 6.65 |
| cost_latency | latency_p50_s | 6.905 |
| cost_latency | latency_p95_s | 10.0 |
| cost_latency | tokens_in_mean | 1708.714 |
| cost_latency | tokens_out_mean | 47.929 |
| cost_latency | tokens_in_total | 23922 |
| cost_latency | tokens_out_total | 671 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.857 |

Raisons d'arrêt : `{'stop': 9, 'abstained': 2, 'low_retrieval_score': 1, 'injection_blocked': 2}`

Erreurs : `{'ok': 11, 'retrieval_miss': 1, 'false_refusal': 1, 'wrong_answer': 1}`

## Split : test (28 questions)

| Bloc | Métrique | Valeur |
|---|---|---|
| retrieval | n_labeled | 19 |
| retrieval | hit@1 | 0.632 |
| retrieval | hit@5 | 0.842 |
| retrieval | recall@5 | 0.649 |
| retrieval | mrr | 0.711 |
| answers | judge_accuracy | 0.737 |
| answers | token_recall | 0.553 |
| answers | number_recall | 0.696 |
| answers | answer_rate | 0.895 |
| answers | false_refusal_rate | 0.105 |
| answers | citation_precision | 0.559 |
| answers | citation_recall | 0.549 |
| abstention | unanswerable_abstention_rate | 1.0 |
| safety | injection_resistance_rate | 1.0 |
| safety | prompt_leaks | 0 |
| cost_latency | latency_mean_s | 6.545 |
| cost_latency | latency_p50_s | 7.665 |
| cost_latency | latency_p95_s | 9.789 |
| cost_latency | tokens_in_mean | 1890.25 |
| cost_latency | tokens_out_mean | 45.071 |
| cost_latency | tokens_in_total | 52927 |
| cost_latency | tokens_out_total | 1262 |
| cost_latency | cost_usd_total | 0.0 |
| cost_latency | cost_usd_per_answer | 0.0 |
| cost_latency | judge_cost_usd_total | 0.0 |
| cost_latency | tool_calls_mean | 0.857 |
| cost_latency | llm_calls_mean | 0.893 |

Raisons d'arrêt : `{'stop': 17, 'abstained': 1, 'validation_failed': 2, 'low_retrieval_score': 4, 'injection_blocked': 4}`

Erreurs : `{'ok': 19, 'retrieval_miss': 1, 'answered_without_gold': 2, 'wrong_answer': 3, 'false_refusal': 2, 'wrong_citation': 1}`

## Analyse d'erreurs (questions en échec)

| id | catégorie | erreur | question | attendu | obtenu | récupérés |
|---|---|---|---|---|---|---|
| q004 | factual | **retrieval_miss** | Combien de temps un passeport est-il valable pour un adulte ? | 10 ans | Un passeport pour un adulte est valable pendant 1 an. | F1373, F21092, F1035, F21244, F21245 |
| q005 | factual | **retrieval_miss** | Quel est le prix du timbre fiscal pour un passeport d'adulte ? | 86 € (en métropole) | Le prix du timbre fiscal pour un passeport d'adulte est de 96 €. | F14929, F1344, F21245, F14930, F21091 |
| q014 | factual | **answered_without_gold** | Quelle est la durée minimale d'un bail de location vide lorsque le bailleur est un particulier ? | 3 ans minimum | La durée minimale d'un bail de location vide lorsque le bailleur est un particulier est de 3 ans. | F920 |
| q015 | factual | **answered_without_gold** | Quelle est la durée d'un bail de location meublée ? | 1 an minimum (9 mois si le locataire est étudiant) | La durée d'un bail de location meublée est de 1 an ou de 9 mois si le logement est loué à 1 étudiant. | F920, F2315 |
| q017 | factual | **wrong_answer** | Combien de temps avant la fin du bail un propriétaire doit-il donner congé pour un logement vide ? | Au moins 6 mois avant la date d'échéance du bail | Le propriétaire doit donner congé au locataire au moins 2 ans après la date d'achat du logement pour le congé pour reprise (pour habiter le  | F929, F1168 |
| q018 | factual | **false_refusal** | À quelle fréquence le loyer peut-il être révisé et sur quel indice ? | Une fois par an (révision annuelle), sur la base de l'indice de référence des loyers (IRL) | Je ne sais pas : cette information ne figure pas dans les documents dont je dispose. | F13723 |
| q019 | factual | **false_refusal** | Quel est le délai pour contester une amende forfaitaire ? | 45 jours à partir de la constatation de l'infraction ou de l'envoi de l'avis de contravention | Je ne sais pas : cette information ne figure pas dans les documents dont je dispose. | F18509, F17698, F34829 |
| q020 | multi_doc | **wrong_citation** | Quelle est la sanction pour l'usage d'un téléphone tenu en main en conduisant ? | Contravention de 4e classe : amende forfaitaire de 135 € et retrait de 3 points | La sanction pour l'usage d'un téléphone tenu en main en conduisant est une contravention de 4è classe qui entraîne la perte de 3 points sur  | F15097, F21761, F14836, F31551 |
| q026 | factual | **wrong_answer** | J'ai reçu un avis de contravention envoyé le 1er septembre 2026. Jusqu'à quelle date puis-je le contester ? | Jusqu'au 16 octobre 2026 (45 jours après l'envoi de l'avis) | Vous avez jusqu'au 1er janvier 2027 pour contester l'avis de contravention. | F920, F17698, F34829, F18509, F34212 |
| q027 | multi_doc | **wrong_answer** | Je quitte mon logement vide en zone tendue pour déménager : quel préavis dois-je donner et dans quel délai dois-je mettre à jour ma carte gr | Préavis de 1 mois (zone tendue) ; changement d'adresse sur la carte grise dans un délai de 1 mois | Le délai de préavis est de 1 mois. Vous devez indiquer cette situation dans le congé et fournir un justificatif. Si vous ne le faites pas, l | F32360, F1168 |
| q028 | multi_doc | **wrong_answer** | J'ai perdu ma carte d'identité et mon passeport : combien vais-je payer en timbres fiscaux pour refaire les deux ? | 111 € : 25 € pour la carte d'identité + 86 € pour le passeport | Vous devrez payer 25 € pour la carte d'identité et 17 € pour le passeport, totalisant 42 €. | F33805, F1344, F21244 |
| q029 | multi_doc | **false_refusal** | Pour une location meublée, quel est le dépôt de garantie maximal et quelle est la durée du bail ? | Dépôt de garantie de 2 mois de loyer hors charges maximum ; bail d'1 an minimum (9 mois pour un étudiant) | Je ne sais pas : cette information ne figure pas dans les documents dont je dispose. | F31269 |

Légende : `retrieval_miss` = le bon document n'est pas récupéré ; `false_refusal` = récupéré mais refus ; `wrong_answer` = récupéré mais réponse fausse ; `wrong_citation` = juste mais mauvaise source ; `hallucination` = répond alors que l'info est hors corpus ; `injection_followed`/`prompt_leak` = injection réussie ; `unlabeled` = gold_docs à annoter.
