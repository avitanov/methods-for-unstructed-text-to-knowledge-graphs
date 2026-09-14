# Supplementary Table S1. Search protocol and candidate-record disposition

The search covered literature available through August 2026. Elicit supplied 24 candidate records, Connected Papers supplied five additional records, and controlled one-hop OpenAlex expansion supplied 253 unique records. Records were consolidated by bibliographic identifier, DOI, and normalized title.

## Elicit search formulations

“Carried forward” denotes records entered into the candidate set after relevance assessment within the search output.

| No. | Search focus | Search formulation | Records carried forward |
| ---: | --- | --- | ---: |
| 1 | Broad review | systematic review of methods for constructing knowledge graphs from unstructured text | 0 |
| 2 | Surveys | What surveys and systematic reviews describe methods for constructing knowledge graphs from unstructured natural-language text? | 4 |
| 3 | End-to-end construction | What methods construct knowledge graphs end-to-end from unstructured text? | 1 |
| 4 | Rules, patterns, dependencies, and OpenIE | Which original research papers introduce rule-based, pattern-based, dependency-parsing, or Open Information Extraction methods for extracting entities, relations, or triples from unstructured text for knowledge graph construction? Exclude surveys, GraphRAG, knowledge graph question answering, and papers that only use an existing knowledge graph. | 2 |
| 5 | Classical machine learning | Which original research papers use classical machine-learning methods such as CRF, SVM, probabilistic models, or feature-based relation extraction to transform unstructured text into entities, relations, triples, or knowledge graphs? Exclude surveys and papers without direct Text-to-Knowledge-Graph relevance. | 1 |
| 6 | Neural methods | Which original research papers use neural networks such as CNNs, RNNs, BiLSTMs, or neural sequence-labeling models to extract entities and relations from unstructured text for knowledge graph construction? Exclude surveys, GraphRAG, knowledge graph question answering, and papers that only use an existing knowledge graph. | 1 |
| 7 | Transformer encoders | Which original research papers use transformer encoder models such as BERT for entity extraction, relation extraction, or joint entity-relation extraction from unstructured text for knowledge graph construction? Prioritize methods whose extracted entities and relations directly populate a knowledge graph. Exclude surveys and papers without direct Text-to-Knowledge-Graph relevance. | 2 |
| 8 | Joint extraction | Which original research papers jointly extract entities and relations from unstructured text for knowledge graph construction? Prioritize methods that reduce pipeline error propagation and produce entity-relation triples or graph structures. Exclude surveys, GraphRAG, knowledge graph question answering, and papers that only use an existing knowledge graph. | 1 |
| 9 | Generative methods | Which original research papers use generative sequence-to-sequence, text-to-triple, or text-to-graph methods to transform unstructured natural-language text into structured knowledge graph output? Include methods that generate entities, relations, triples, RDF, or graph structures. Exclude surveys and papers without direct Text-to-Knowledge-Graph relevance. | 2 |
| 10 | Prompted LLMs | Which original research papers use zero-shot or few-shot large language model prompting to construct knowledge graphs from unstructured natural-language text? Prioritize methods that extract entities, relations, triples, RDF, or complete graph structures directly from text. Exclude surveys, GraphRAG, knowledge graph question answering, and papers that only use an existing knowledge graph. | 1 |
| 11 | LLM pipelines | Which original research papers use end-to-end or incremental pipelines that use large language models to construct, integrate, or update knowledge graphs from unstructured documents? Prioritize systems that handle multiple stages such as entity extraction, relation extraction, entity resolution, triple generation, or graph integration. Exclude GraphRAG and systems that only query existing knowledge graphs. | 1 |
| 12 | Fine-tuned LLMs | Which original research papers use instruction-tuned or fine-tuned large language models to transform unstructured text into knowledge graph triples or graph structures? Include supervised fine-tuning, instruction tuning, parameter-efficient fine-tuning, and schema-conditioned generation. Exclude surveys and papers without direct Text-to-Knowledge-Graph construction. | 1 |
| 13 | Incremental construction | Which original research papers present end-to-end or incremental pipelines that use large language models to construct, integrate, or update knowledge graphs from unstructured documents? Prioritize systems that handle multiple stages such as entity extraction, relation extraction, entity resolution, triple generation, or graph integration. Exclude GraphRAG and systems that only query existing knowledge graphs. | 1 |
| 14 | Ontology and schema guidance | Which original research papers use ontologies, schemas, or controlled relation vocabularies to guide the construction of knowledge graphs from unstructured natural-language text? Include NLP, transformer, and LLM methods that constrain entity types, relations, triples, RDF, or graph output. Exclude surveys, GraphRAG, knowledge graph question answering, and papers that only consume an existing knowledge graph. | 1 |
| 15 | Hybrid methods | Which original research papers combine traditional NLP or transformer components with large language models to construct knowledge graphs from unstructured text? Prioritize pipelines that divide tasks such as entity extraction, relation extraction, entity linking, graph construction, or validation between multiple techniques. Exclude surveys and systems without direct Text-to-Knowledge-Graph construction. | 1 |
| 16 | Validation and integration | Which original research papers validate, correct, deduplicate, link, or integrate knowledge graph triples extracted from unstructured text? Include ontology verification, schema validation, entity linking, confidence estimation, hallucination reduction, and knowledge fusion. Exclude papers that only perform knowledge graph completion without extracting information from text. | 1 |
| 17 | Benchmarks | Which original research papers introduce benchmarks or datasets for evaluating systems that transform unstructured natural-language text into knowledge graph triples or graph structures? Prioritize benchmarks that evaluate entity extraction, relation extraction, triple generation, RDF generation, or complete Text-to-Knowledge-Graph construction. Exclude surveys, GraphRAG, knowledge graph question answering, and knowledge graph completion benchmarks without text input. | 2 |
| 18 | Evaluation methods | Which original research papers propose evaluation methods for generative or large-language-model-based knowledge graph construction from text? Include evaluation of triple correctness, factual grounding, hallucination, ontology compliance, schema validity, duplicate entities, and graph quality. Exclude papers that only report ordinary relation-extraction F1 without proposing a benchmark or evaluation methodology. | 1 |
| 19 | End-to-end graph quality | Which original research papers evaluate end-to-end knowledge graph construction from unstructured text beyond entity-level or relation-level F1? Prioritize methods that measure graph-level quality, factual consistency, provenance, schema compliance, or human evaluation. | 0 |

## Connected Papers seed expansion

Connected Papers graphs were examined for the following seed records.

| Seed record | Records carried forward |
| --- | ---: |
| *A Comprehensive Survey on Automatic Knowledge Graph Construction* | 1 |
| *OpenIE-based Approach for Knowledge Graph Construction from Text* | 3 |
| *Joint Extraction of Entities and Relations Based on a Novel Graph Scheme* | 0 |
| *Seq2KG: An End-to-End Neural Model for Domain Agnostic Knowledge Graph Construction from Text* | 1 |

## OpenAlex one-hop expansion

OpenAlex expansion followed a non-recursive, one-hop design based on provisionally eligible seed-stage records. For each bibliographically matched origin, the process inspected up to 100 referenced works and retained no more than five after relevance ranking; inspected up to ten highly cited and ten recent citing works and retained no more than five; and inspected up to 20 related works and retained no more than five.

| Expansion stage | Records |
| --- | ---: |
| Origin–relationship observations before consolidation | 309 |
| Duplicates against the seed set | 8 |
| Unique OpenAlex expansion records after consolidation | 253 |
| Records advanced to title-and-abstract screening | 150 |
| Records not advanced after relevance ranking | 103 |

## Selection summary

| Selection stage | Records |
| --- | ---: |
| Total records in the triage pool | 282 |
| Records screened by title and abstract | 179 |
| Records excluded during title-and-abstract screening | 64 |
| Records entering final eligibility assessment | 115 |
| Records not included after eligibility and overlap assessment | 32 |
| Full-text papers included in the synthesis | 83 |
