# Human check of the LLM judge

10 answers of the final run (`eval/results.jsonl`) drawn at random (seed 0) among the answerable questions.
For each one: read the reference answer and the system answer, write down your own verdict
(correct / partial / incorrect), then say whether you agree with the judge.
Agreement = number of "yes" rows / 10, to report in `docs/evaluation.md` and at the defence.
Questions, references and answers are left in French, as produced by the system.

| id | Question | Reference | System answer | Judge | Human verdict | Agreement (yes/no) |
|---|---|---|---|---|---|---|
| q002 | Combien coûte le remplacement d'une carte d'identité perdue ou volée ? | 25 € en timbre fiscal (12,50 € en Guyane) | Le renouvellement d’une carte d’identité qui a été perdue ou volée coûte 25 € en France métropolitaine et en Guyane. En Martinique, Guadeloupe, La Réunion et Mayotte, le coût est de 12,50 €. | correct |  |  |
| q009 | Quel est le montant maximal du dépôt de garantie pour une location vide ? | 1 mois de loyer hors charges | Le montant maximal du dépôt de garantie pour une location vide ne doit pas dépasser 1 mois de loyer (hors charges). | correct |  |  |
| q010 | Quel est le montant maximal du dépôt de garantie pour une location meublée ? | 2 mois de loyer hors charges | Le montant maximal du dépôt de garantie pour une location meublée ne doit pas dépasser 2 mois de loyer (hors charges). | correct |  |  |
| q013 | Quel est le préavis du locataire d'un logement vide situé en zone tendue ? | 1 mois | Le préavis du locataire d'un logement vide situé en zone tendue est de 1 mois, à condition d'indiquer dans la lettre de congé l'adresse du logement loué et le fait de bénéficier d'un délai de préavis réduit à un mois. | correct |  |  |
| q014 | Quelle est la durée minimale d'un bail de location vide lorsque le bailleur est un particulier ? | 3 ans minimum | La durée minimale d'un bail de location vide lorsque le bailleur est un particulier est de 3 ans. | correct |  |  |
| q016 | À quelle période s'applique la trêve hivernale des expulsions locatives ? | Du 1er novembre au 31 mars de l'année suivante | La trêve hivernale s'applique du 1er novembre au 31 mars de l'année suivante. L'expulsion du locataire est reportée pendant cette période. | correct |  |  |
| q017 | Combien de temps avant la fin du bail un propriétaire doit-il donner congé pour un logement vide ? | Au moins 6 mois avant la date d'échéance du bail | Le propriétaire doit donner congé pour un logement vide au plus tard 2 ans avant la date d'échéance du bail en cours. | incorrect |  |  |
| q025 | Quel est le montant d'une amende forfaitaire minorée de 4e classe ? | 90 € | Le montant d'une amende forfaitaire minorée de 4e classe est de 90 €. | correct |  |  |
| q028 | J'ai perdu ma carte d'identité et mon passeport : combien vais-je payer en timbres fiscaux pour refaire les deux ? | 111 € : 25 € pour la carte d'identité + 86 € pour le passeport | Vous devrez payer 50 € en timbres fiscaux pour renouveler à la fois votre carte d'identité et votre passeport, car chaque document coûte 25 €. | incorrect |  |  |
| q029 | Pour une location meublée, quel est le dépôt de garantie maximal et quelle est la durée du bail ? | Dépôt de garantie de 2 mois de loyer hors charges maximum ; bail d'1 an minimum (9 mois pour un étudiant) | Je ne sais pas : cette information ne figure pas dans les documents dont je dispose. | incorrect |  |  |

**Agreement: __ / 10** — checked by: ____
