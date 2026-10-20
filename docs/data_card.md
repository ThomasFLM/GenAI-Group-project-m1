# Data card — Service-Public.fr "Particuliers" sub-corpus

> Figures produced by `uv run python -m rag.cli stats` (file `docs/corpus_stats.json`).

## Source

| | |
|---|---|
| Dataset | "Fiches pratiques et ressources de Service-Public.fr Particuliers" (data.gouv.fr) |
| Producer | DILA — Direction de l'information légale et administrative (French government legal information office) |
| Format | ZIP archive of XML files (`vosdroits-latest.zip`), one fact sheet per file, declaring the XSD schema 3.5 (copy in `docs/schema/3.5/`, root element `Publication`) |
| Licence | Licence Ouverte / Open Licence 2.0 (Etalab) — to be checked on the dataset page |
| Download date | 13/10/2026 (`vosdroits-latest.zip`, 24 MB, 145 MB uncompressed) |
| Alternative | DINUM "vectorised" export (Parquet, Hugging Face): loadable with `build-corpus --source file.parquet`. Only the text is used; our own chunking and embeddings; their embeddings can serve as a baseline. |

The fact sheets are written in French; theme names, fact-sheet types and XML element names are quoted
as they appear in the data.

## Selection (~500 documents)

The archive downloaded on 13/10/2026 contains 5,573 XML files: 3,003 fact sheets `F*`, 2,318 resources `R*`
(forms, websites, addresses), 243 folders/themes `N*` and a few indexes. The fact sheets declare schema
3.5 (`xsi:noNamespaceSchemaLocation="../Schemas/3.5/Publication.xsd"`) but **only 634 out of 3,003 actually
pass** validation (lxml): most deviations come from a `TitreRiche` element missing from the schema
(6,438 occurrences, a duplicate of the case title, ignored by the parser), then from `Liste`, `BlocCas`,
`ServiceEnLigne`, `ASavoir` placed where the schema does not expect them. The parser therefore walks the
tree generically instead of strictly following the schema.

| Fact-sheet type (`@type`) | Count | Words (median) | Kept? |
|---|---|---|---|
| Fiche d'information conditionnée (information sheet) | 1,496 | 1,904 | yes |
| Fiche Question-réponse conditionnée (Q&A sheet) | 1,385 | 686 | yes |
| Recherche guidée (guided search) | 85 | 60 (max 858,552) | no: decision-tree questionnaire, not a text |
| Fiche Comment faire si conditionné ("what to do if") | 22 | 3,088 | no: aggregate of other fact sheets |
| Fiche avec liens externes (external links) | 15 | 48 | no: links only |

Information + Q&A fact sheets: 2,881, spread over 11 themes (Travail 621, Argent 394, Famille 377,
Logement 350, Social 263, Justice 233, Transports 214, Papiers 199, Étranger 173, ...).

Themes kept: **Papiers - Citoyenneté - Élections** (identity papers, citizenship, elections), **Logement**
(housing), **Transports - Mobilité** (transport) — 729 eligible fact sheets. Why: frequent questions from
private individuals, factual and checkable answers (deadlines, amounts, durations), and many close cases
(unfurnished / furnished rental, adult / minor, mainland France / French Guiana) that make retrieval
non-trivial. Selection of 500 fact sheets: even spread across themes, deterministic order by identifier,
the 30 golden-set fact sheets are always included.
Deliberately out of corpus: tax, work, social benefits, family, businesses → "je ne sais pas" questions.

## Statistics (kept corpus, `docs/corpus_stats.json`)

| Measure | Value |
|---|---|
| Fact sheets kept | 500 (269 information sheets, 231 Q&A sheets) |
| Split by theme | Logement 168 · Papiers 166 · Transports 166 |
| Words per fact sheet | mean 2,255 · median 1,389 · min 30 · max 44,883 · total 1.13 M |
| Chunks after chunking (220 words, 40 overlap) | 8,297 |
| Words per chunk | mean 159 · median 195 · min 4 · max 261 |
| Update dates | 2021-06-22 → 2026-10-13; 495 of 500 fact sheets updated since 2024 |

## Cleaning and observations

- Removed: breadcrumbs, menus, parent folders (lists of sibling fact sheets), "Voir aussi" (see also),
  references, online services, "Où s'adresser" (where to go), media, personalisation logic (`Condition`),
  alternative titles.
- "Conditionnée" fact sheets contain **all variants** (France / French Guiana / abroad, adult / minor,
  unfurnished / furnished...): hence very long fact sheets (up to 45,000 words) and cases nested 5 levels
  deep. Consequence: the case heading is kept in every chunk ("### En Guyane").
- Tables (`<Rangée>`) linearised with their headers: "Contravention : 4e classe ; Minoré : 90 €".
- Pitfalls met while reading 20 fact sheets and 20 chunks (`docs/samples_to_read.md`), and fixed:
  spurious spaces around inline tags ("1 er", "6 e vendredi"), then lost spaces ("validitéde");
  duplicated case titles (`TitreRiche`); crumbs of 1 to 40 words with the first section-based chunking
  (14,632 chunks, median 41 words) → grouping by chapter (8,297, median 195).
- Remaining: sentences that point to a removed link ("sur la plateforme suivante :" — on the following
  platform), very long case paths.

## Limitations and risks

- Amounts and deadlines change (finance laws): the corpus is a dated snapshot; the assistant may give
  outdated information if the archive is not refreshed. The fact sheet's date is exposed in the context
  and the metadata.
- Fact sheets often point to simulators or to special cases not covered by the text.
- The corpus is written in administrative French; colloquial questions ("caution", "PV") depend on the
  dense retriever.
