# Architecture et choix techniques

Chaque choix ci-dessous est une **hypothèse à mesurer** sur le split `dev` du golden set, puis confirmée
une seule fois sur le split `test` (questions tenues à l'écart pendant le réglage).

## 1. Ingestion et nettoyage (`ingestion.py`)

- Source : archive XML officielle `vosdroits-latest.zip` (DILA), une fiche = un fichier `F*.xml`.
  Identifiant = attribut `ID` (ex. `F1234`), réutilisé tel quel dans `gold_docs` et dans les citations.
- On garde : titre, résumé, thème, date de modification, et le texte structuré (chapitres, sous-chapitres,
  cas/situations, listes, tableaux, encadrés « À savoir / Attention »).
- On retire : fil d'Ariane, menus, « Voir aussi », références juridiques, services en ligne, « Où s'adresser »,
  partenaires — du bruit pour la recherche lexicale qui ne contient pas de réponse.
- Normalisation : Unicode NFC, espaces insécables, espaces multiples.
- Sélection (~500) : thèmes *Papiers - Citoyenneté - Élections*, *Logement*, *Transports - Mobilité*,
  répartis équitablement, ordre déterministe ; les fiches du golden set sont toujours incluses.

## 2. Découpage (`chunking.py`)

- **Par section** : un passage ne mélange jamais deux sections. Les fiches sont organisées en cas
  (« location vide » / « location meublée », « majeur » / « mineur ») ; mélanger deux cas dans un passage
  fait répondre le modèle avec le mauvais chiffre.
- Sections longues : regroupement de paragraphes jusqu'à **220 mots**, **40 mots** de recouvrement.
  (≈ 300 tokens : assez pour une règle et ses exceptions, assez court pour que 5 passages tiennent
  dans le contexte d'un modèle 7B.)
- **En-tête contextuel** indexé avec chaque passage : « titre de la fiche — chemin de section ».
  Sans lui, un passage « Le délai est de 2 mois » n'est retrouvable ni par BM25 ni par les embeddings.
- Expériences prévues : 120 / 220 / 400 mots, avec/sans en-tête → hit@5 et MRR sur `dev`.

## 3. Recherche hybride (`retrieval.py`)

| Étape | Choix | Pourquoi |
|---|---|---|
| Lexical | BM25 (rank-bm25), tokenisation française sans accents, mots vides, racinisation légère | les questions contiennent des termes exacts (« dépôt de garantie », « zone tendue ») |
| Dense | `multilingual-e5-small` (préfixes `query:` / `passage:`), cosinus | paraphrases (« rendre la caution » ↔ « restitution du dépôt de garantie ») |
| Fusion | RRF (k = 60), 30 candidats par moteur | robuste, ne demande pas de normaliser des scores hétérogènes |
| Reranking | cross-encoder multilingue sur les 30 candidats → top 5 | précision du top 5, et score absolu utilisable comme seuil d'abstention |
| Filtres | date de mise à jour (`since`), thème | questions « depuis 2024 », « dernières fiches » |

Index en mémoire (numpy) : ~3 000 passages × 384 dimensions, recherche < 10 ms ; une base vectorielle
n'apporterait rien à cette échelle (piste bonus : Qdrant pour les filtres de métadonnées).

## 4. Génération (`pipeline.py`, `prompts.py`, `llm.py`)

- LLM local via Ollama, température 0, `seed` fixé (reproductibilité).
- **Sortie structurée** : schéma JSON imposé à Ollama (`format`), puis validation pydantic :
  `{"answerable": bool, "answer": str, "citations": [str]}`. Citations hors des documents récupérés
  supprimées ; réponse « answerable » sans citation valide = invalide → **un** nouvel essai avec le message
  d'erreur, puis refus standard.
- **Abstention** à deux niveaux : (1) score du reranker sous le seuil → refus sans appeler le LLM
  (économise latence et tokens) ; (2) le LLM met `answerable: false`.
- **Mode agent** : boucle d'appels d'outils (≤ 4 étapes) — `search_corpus` (reformulation, multi-requêtes,
  filtre de date), `get_document` (lecture d'une fiche complète), `add_days` (calcul de date limite :
  les LLM se trompent en arithmétique de dates). Si l'agent ne cherche rien, une recherche est imposée.
- **Mesure** : tokens in/out (`prompt_eval_count` / `eval_count` d'Ollama, `usage` de l'API), latence
  par étape (BM25, dense, rerank, LLM), appels d'outils, coût ($ API ; 0 $ en local, coût matériel discuté
  à part), raison d'arrêt (`stop`, `length`, `abstained`, `low_retrieval_score`, `injection_blocked`,
  `validation_failed`, `output_blocked_leak`, `max_steps`, `llm_error`).

## 5. Sécurité : injections de prompt (`guardrails.py`)

Défense en profondeur :
1. **Entrée** : motifs FR/EN (« ignore les instructions », « prompt système », « tu es désormais »,
   balises `<system>`…). Politique `block` (refus immédiat) ou `warn` (on laisse passer, le reste protège).
2. **Contexte** : passages et question encadrés par des balises et déclarés comme *données*.
   Injection indirecte testée : une fiche de test contient « Ignore les instructions précédentes ».
3. **Sortie** : canari secret dans le prompt système ; s'il ressort, la réponse est bloquée. Les citations
   sont limitées aux documents réellement récupérés (pas de fausse source).

Limite assumée : les motifs ont été écrits en voyant les injections du golden set → le taux de résistance
mesuré est optimiste. Pour une mesure honnête, ajouter des injections nouvelles dans le split `test`
sans retoucher les motifs (cf. evaluation.md).
