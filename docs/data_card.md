# Data card — sous-corpus Service-Public.fr Particuliers

> Chiffres produits par `uv run python -m rag.cli stats` (fichier `docs/corpus_stats.json`).

## Source

| | |
|---|---|
| Jeu de données | « Fiches pratiques et ressources de Service-Public.fr Particuliers » (data.gouv.fr) |
| Producteur | DILA — Direction de l'information légale et administrative |
| Format | archive ZIP de fichiers XML (`vosdroits-latest.zip`), une fiche par fichier, conformes au schéma XSD 3.5 (copie dans `docs/schema/3.5/`, élément racine `Publication`) |
| Licence | Licence Ouverte / Open Licence 2.0 (Etalab) — à vérifier sur la page du jeu |
| Date de téléchargement | 13/10/2026 (`vosdroits-latest.zip`, 24 Mo, 145 Mo décompressés) |
| Alternative | export « vectorisé » de la DINUM (Parquet, Hugging Face) : chargeable via `build-corpus --source fichier.parquet`. Seul le texte est utilisé ; nos propres découpage et embeddings ; leurs embeddings peuvent servir de baseline. |

## Sélection (~500 documents)

L'archive téléchargée le 13/10/2026 contient 5 573 fichiers XML : 3 003 fiches `F*`, 2 318 ressources `R*`
(formulaires, sites, adresses), 243 dossiers/thèmes `N*` et quelques index. Les fiches déclarent le schéma
3.5 (`xsi:noNamespaceSchemaLocation="../Schemas/3.5/Publication.xsd"`) mais **seules 634 sur 3 003 le
respectent** à la validation (lxml) : la plupart des écarts viennent d'un élément `TitreRiche` absent du
schéma (6 438 occurrences, doublon du titre des cas, ignoré par le parseur), puis de `Liste`, `BlocCas`,
`ServiceEnLigne`, `ASavoir` placés à des endroits non prévus. Le parseur parcourt donc l'arbre de façon
générique au lieu de suivre strictement le schéma.

| Type de fiche (`@type`) | Nb | Mots (médiane) | Retenu ? |
|---|---|---|---|
| Fiche d'information conditionnée | 1 496 | 1 904 | oui |
| Fiche Question-réponse conditionnée | 1 385 | 686 | oui |
| Recherche guidée | 85 | 60 (max 858 552) | non : questionnaire arborescent, pas un texte |
| Fiche Comment faire si conditionné | 22 | 3 088 | non : agrégat d'autres fiches |
| Fiche avec liens externes | 15 | 48 | non : uniquement des liens |

Fiches d'information + questions-réponses : 2 881, réparties en 11 thèmes (Travail 621, Argent 394,
Famille 377, Logement 350, Social 263, Justice 233, Transports 214, Papiers 199, Étranger 173, ...).

Thèmes retenus : **Papiers - Citoyenneté - Élections**, **Logement**, **Transports - Mobilité**
(729 fiches éligibles). Pourquoi : questions fréquentes des particuliers, réponses factuelles et vérifiables
(délais, montants, durées), beaucoup de cas proches (location vide / meublée, majeur / mineur, métropole /
Guyane) qui rendent le retrieval non trivial. Sélection de 500 fiches : répartition équilibrée entre thèmes,
ordre déterministe par identifiant, les 30 fiches du golden set sont toujours incluses.
Hors corpus volontairement : fiscalité, travail, social, famille, entreprises → questions « je ne sais pas ».

## Statistiques (corpus retenu, `docs/corpus_stats.json`)

| Mesure | Valeur |
|---|---|
| Fiches retenues | 500 (269 fiches d'information, 231 questions-réponses) |
| Répartition par thème | Logement 168 · Papiers 166 · Transports 166 |
| Mots par fiche | moyenne 2 255 · médiane 1 389 · min 30 · max 44 883 · total 1,13 M |
| Passages après découpage (220 mots, recouvrement 40) | 8 297 |
| Mots par passage | moyenne 159 · médiane 195 · min 4 · max 261 |
| Dates de mise à jour | 2021-06-22 → 2026-10-13 ; 495 fiches sur 500 mises à jour depuis 2024 |

## Nettoyage et observations

- Supprimé : fil d'Ariane, menus, dossiers parents (listes de fiches sœurs), « Voir aussi », références,
  services en ligne, « Où s'adresser », médias, logique de personnalisation (`Condition`), titres alternatifs.
- Les fiches « conditionnées » contiennent **toutes les variantes** (France / Guyane / étranger, majeur /
  mineur, vide / meublé...) : d'où des fiches très longues (jusqu'à 45 000 mots) et des cas imbriqués sur
  5 niveaux. Conséquence : le titre du cas est gardé dans chaque passage (« ### En Guyane »).
- Tableaux (`<Rangée>`) linéarisés avec les en-têtes : « Contravention : 4e classe ; Minoré : 90 € ».
- Pièges rencontrés en lisant 20 fiches et 20 passages (`docs/samples_to_read.md`) et corrigés :
  espaces parasites autour des balises inline (« 1 er », « 6 e vendredi ») puis espaces perdus
  (« validitéde ») ; titres de cas en double (`TitreRiche`) ; miettes de 1 à 40 mots avec le premier
  découpage par section (14 632 passages, médiane 41 mots) → regroupement par chapitre (8 297, médiane 195).
- Restent : phrases qui renvoient à un lien supprimé (« sur la plateforme suivante : »), chemins de cas très
  longs.

## Limites et risques

- Les montants et délais changent (lois de finances) : le corpus est un instantané daté ; l'assistant
  peut donner une information périmée si l'archive n'est pas mise à jour. La date de la fiche est exposée
  dans le contexte et les métadonnées.
- Les fiches renvoient souvent à des simulateurs ou à des cas particuliers non couverts par le texte.
- Corpus en français administratif ; les questions familières (« caution », « PV ») dépendent du dense.
