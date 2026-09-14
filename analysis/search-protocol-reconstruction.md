# Search protocol reconstruction

## Scope and evidence

This audit reconstructs the literature-search process from contemporaneous Chrome history, surviving Elicit sessions, retrievable Connected Papers graph snapshots, and Codex session records. Only activity involving Elicit, Connected Papers, and OpenAlex was inspected. Authentication pages and unrelated browsing or session content were excluded.

The recovered chronology is:

- 31 July–2 August 2026: Elicit and Connected Papers discovery.
- 2–3 August 2026: OpenAlex metadata verification and controlled one-hop expansion.
- 3 August 2026: title-and-abstract screening and consolidation.
- 3–13 August 2026: full-text retrieval, eligibility assessment, and final inclusion.

The exact query and result evidence is preserved in `data/review/search-query-log.csv`. The 29 records entered during seed discovery are preserved in `data/review/discovery-records.csv`. The three browser-verified Connected Papers graphs are preserved at result level in `data/review/connected-papers-graph-results.csv`.

## Elicit discovery

Nineteen distinct natural-language query formulations were recovered. Seventeen were associated with the 24 records entered into the initial candidate log; one broader query launched the discovery process, and one later query examined end-to-end evaluation coverage without adding another seed record.

The queries systematically covered surveys, rule- and pattern-based methods, classical machine learning, neural models, transformer encoders, joint extraction, generative text-to-structure models, prompted and fine-tuned LLMs, ontology- or schema-guided methods, hybrid systems, validation and integration, benchmarks, and graph-level evaluation. They consistently required unstructured natural-language input and direct knowledge-graph-compatible output, while excluding GraphRAG-only, KG-question-answering, completion-only, and existing-KG-only work.

Six surviving Elicit outputs expose numerical result information:

- ontology-guided construction: 96 sources screened and 22 records curated;
- hybrid NLP–LLM construction: 13 table rows;
- validation and integration: 15 table rows;
- benchmark and dataset papers: 10 table rows;
- generative/LLM evaluation methods: 8 table rows;
- end-to-end graph-quality evaluation: 80 sources surveyed and 10 table rows.

The remaining Elicit output counts are not recoverable. Consequently, 24 is the documented number of Elicit records entered into the candidate log, not the number of raw platform hits. Elicit result sets overlapped and were screened semantically before records were entered.

## Connected Papers expansion

Chrome history verifies three graph origins created on 2 August 2026:

- *A Comprehensive Survey on Automatic Knowledge Graph Construction*;
- *OpenIE-based Approach for Knowledge Graph Construction from Text*;
- *Joint Extraction of Entities and Relations Based on a Novel Graph Scheme*.

Each surviving snapshot contains 41 nodes. The three graphs contain 123 unique result nodes in total, with no overlap between the snapshots. Nine of those graph nodes are among the final 83 studies.

The contemporaneous candidate-entry record documents five records added from Connected Papers: one from the broad-survey graph, three from the OpenIE graph, and one attributed to a Seq2KG graph. The Seq2KG graph visit is not present in the retained Chrome history, so its full result count cannot be reconstructed. The five entered records are nevertheless documented by title and origin in `data/review/discovery-records.csv`.

## OpenAlex verification and expansion

OpenAlex was first used to verify bibliographic metadata for the 29 seed-stage records. It was then used for controlled one-hop expansion from the 28 records provisionally included after initial screening; the one uncertain record was not used as an origin.

Origin resolution followed this order: existing OpenAlex identifier, DOI lookup, exact-title lookup, and title similarity of at least 0.96 with a publication-year difference no greater than one. Twenty-six origins were resolved and two remained unresolved.

For every resolved origin, the expansion inspected:

- up to 100 referenced works and retained at most five after local relevance ranking;
- up to ten highly cited and ten recent citing works, then retained at most five;
- up to 20 related works and retained at most five.

Newly discovered records were not used as new origins. Records were deduplicated by OpenAlex identifier, normalized DOI, and normalized title. Duplicate provenance was merged across origins and relation types.

The run retained 309 origin/channel relationship observations before global consolidation. Eight records already present in the 29-record seed set were removed, and repeat observations were merged, leaving 253 unique expansion records. Of these, the 150 highest-ranked records were advanced to title-and-abstract screening and 103 lower-ranked records were not advanced.

## Reconstructed selection flow

The documented counts form the following internally consistent flow:

| Stage | Count |
| --- | ---: |
| Records entered from Elicit outputs | 24 |
| Records added from Connected Papers | 5 |
| Unique OpenAlex expansion records | 253 |
| Total records available for formal triage | 282 |
| Lower-ranked OpenAlex records not advanced | 103 |
| Records screened by title and abstract | 179 |
| Records excluded during title-and-abstract screening | 64 |
| Records retained for full-text assessment | 115 |
| Full texts not retrieved | 31 |
| Full-text assessment unresolved at the cut-off | 1 |
| Full-text studies included in the synthesis | 83 |

The arithmetic is complete: `24 + 5 + 253 = 282`; `282 - 103 = 179`; `179 - 64 = 115`; and `115 - 31 - 1 = 83`.

## Assessment of the query strategy

The recovered queries are logically aligned with the review question. Their strongest feature is explicit coverage of methodology families and an eligibility boundary requiring text input and graph-compatible output. This reduces drift into adjacent KG tasks.

The strategy is not equivalent to a conventional exhaustive database search. Elicit queries are semantic natural-language requests rather than stable Boolean database strings, Connected Papers is seed-dependent, and OpenAlex expansion was capped and locally ranked. Raw result counts were not retained for every Elicit query, and the 103 OpenAlex records below the screening cutoff did not receive title-and-abstract decisions. The manuscript should therefore describe the work as a structured, protocol-guided review with a reproducible documented candidate flow, not as an exhaustive systematic review of every database record.
