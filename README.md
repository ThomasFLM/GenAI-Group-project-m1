# Nom du projet

Assistant IA capable de retrouver des informations dans un corpus de documents et de répondre à des questions à partir de leur contenu.

## Objectif

À partir d'une question posée en langage naturel, le système :

1. retrouve les passages les plus pertinents dans le corpus ;
2. génère une réponse à partir de ces passages ;
3. indique les documents sources utilisés.

## Structure du projet

```
.
├── data/
│   ├── raw/            # documents d'origine (non versionnés si volumineux)
│   └── processed/      # textes nettoyés, découpés en passages
├── src/
│   ├── ingestion/      # lecture et découpage des documents
│   ├── indexing/       # embeddings et index de recherche
│   ├── retrieval/      # recherche des passages pertinents
│   └── generation/     # génération de la réponse
├── notebooks/          # essais et expérimentations
├── tests/
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

## Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

À compléter : commandes pour indexer le corpus et poser une question.

## Équipe

| Nom | Rôle / partie du projet |
|-----|-------------------------|
|     |                         |
|     |                         |

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md).
