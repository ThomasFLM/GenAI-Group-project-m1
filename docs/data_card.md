# Data card — sous-corpus Service-Public.fr Particuliers

> Les chiffres marqués `À REMPLIR` sont produits par `uv run python -m rag.cli stats`
> (fichier `docs/corpus_stats.json`).

## Source

| | |
|---|---|
| Jeu de données | « Fiches pratiques et ressources de Service-Public.fr Particuliers » (data.gouv.fr) |
| Producteur | DILA — Direction de l'information légale et administrative |
| Format | archive ZIP de fichiers XML (`vosdroits-latest.zip`), une fiche par fichier |
| Licence | Licence Ouverte / Open Licence 2.0 (Etalab) — à vérifier sur la page du jeu |
| Date de téléchargement | À REMPLIR |
| Alternative | export « vectorisé » de la DINUM (Parquet, Hugging Face) : chargeable via `build-corpus --source fichier.parquet`. Seul le texte est utilisé ; nos propres découpage et embeddings ; leurs embeddings peuvent servir de baseline. |

## Sélection (~500 documents)

- Type : fiches d'information `F*.xml` uniquement (pas les dossiers `N*`, ni les questions-réponses).
- Thèmes : **Papiers - Citoyenneté - Élections**, **Logement**, **Transports - Mobilité**.
  Pourquoi : questions fréquentes des particuliers, réponses factuelles et vérifiables (délais, montants,
  durées), et beaucoup de cas proches (location vide / meublée, majeur / mineur) qui rendent le retrieval
  non trivial.
- Répartition équilibrée entre thèmes, ordre déterministe par identifiant, fiches du golden set forcées.
- Hors corpus volontairement : fiscalité, travail, social, entreprises → sert aux questions « je ne sais pas ».

## Statistiques (À REMPLIR)

| Mesure | Valeur |
|---|---|
| Fiches lues dans l'archive | |
| Fiches retenues | |
| Répartition par thème | |
| Mots par fiche (moyenne / médiane / min / max) | |
| Passages après découpage | |
| Mots par passage (moyenne / médiane) | |
| Dates de mise à jour (min / max, part depuis 2024) | |
| Fiches quasi vides (< 30 mots) | |

## Nettoyage

- Suppression : fil d'Ariane, menus, « Voir aussi », références de textes, services en ligne, « Où s'adresser ».
- Conservation de la structure (titres de chapitre/cas) dans le champ `section` des passages.
- Normalisation Unicode, espaces insécables, lignes vides.
- Observations après lecture de 20 fiches et 20 passages (`docs/samples_to_read.md`) : À REMPLIR
  (ex. tableaux de montants mal linéarisés ? passages trop courts ? doublons ?).

## Limites et risques

- Les montants et délais changent (lois de finances) : le corpus est un instantané daté ; l'assistant
  peut donner une information périmée si l'archive n'est pas mise à jour. La date de la fiche est exposée
  dans le contexte et les métadonnées.
- Les fiches renvoient souvent à des simulateurs ou à des cas particuliers non couverts par le texte.
- Corpus en français administratif ; les questions familières (« caution », « PV ») dépendent du dense.
