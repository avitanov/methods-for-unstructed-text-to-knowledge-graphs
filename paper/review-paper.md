# From Rules to Large Language Models: A Review of Methods for Constructing Knowledge Graphs from Unstructured Text

## Abstract

Knowledge graphs represent facts as entities and relations, but much useful knowledge remains in ordinary text. Transforming that text into a reliable graph requires more than recognizing names or producing plausible triples. A complete process may need to identify entities, determine their types, resolve repeated mentions, extract relations or events, link mentions to known records, align the output with a schema, merge duplicate facts, and validate the resulting graph. This review synthesizes 83 acquired and analyzed papers on these tasks. It traces the field from rule-based and linguistic systems through classical machine learning, neural and transformer models, joint extraction, generative methods, and large language model (LLM) approaches. The evidence shows that technical progress has mainly changed where constraints are expressed. Early systems encoded constraints in patterns and ontologies; statistical and neural methods learned them from data; current generative systems express them through output formats, prompts, retrieval, and post-generation validation. No methodology family in the reviewed corpus addresses all stages of the defined Text-to-KG pipeline without substantial trade-offs. Rules remain transparent but narrow. Open extraction offers broad coverage but creates noisy and heterogeneous facts. Supervised neural systems are accurate on matched benchmarks but depend on labels and fixed relation inventories. LLMs improve flexibility and reduce task-specific engineering, yet remain sensitive to prompts, output formats, model choice, cost, privacy, and hallucination. The strongest practical designs are therefore hybrid: they combine learned extraction with explicit schemas, entity resolution, deterministic checks, confidence estimation, or human oversight. Evaluation remains fragmented because studies use different datasets, matching rules, graph scopes, and quality criteria. We propose a stage-based framework for comparing methods, identify evidence-supported design principles, and outline priorities for trustworthy, maintainable, and reproducible Text-to-Knowledge-Graph systems.

**Keywords:** knowledge graph construction; information extraction; relation extraction; entity linking; open information extraction; transformers; large language models; ontology; knowledge fusion; graph validation

## 1. Introduction

Large parts of scientific, technical, administrative, and public knowledge are written as natural-language text. Humans can read a sentence and understand who or what it concerns, how the mentioned concepts are related, and whether a later sentence refers to the same thing. Computers require a more explicit representation. A knowledge graph supplies such a representation by storing entities and the relations between them. For example, a sentence stating that a drug treats a disease can become a structured fact linking the drug entity to the disease entity through a `treats` relation.

This conversion appears simple when shown as one sentence and one triple. In practice, it is a chain of dependent decisions. A system must decide which words denote entities, whether two names refer to the same entity, which relation is expressed, whether the relation is allowed by a target ontology, and whether the extracted statement is sufficiently supported by the source. Errors at one stage affect later stages. An incorrect entity boundary can produce an incorrect relation; a correct local triple can still be attached to the wrong real-world entity; and individually plausible triples can form an inconsistent or redundant graph.

The reviewed literature addresses these difficulties from markedly different directions. Rule-based systems describe valid language patterns directly. Open information extraction (OpenIE) systems seek relations without a fixed domain schema. Classical machine-learning systems learn from annotated or distantly supervised examples. Neural and transformer models learn contextual representations and increasingly combine entity and relation extraction. Generative models translate text directly into triples, RDF, or other graph serializations. Recent LLM systems use instructions, demonstrations, fine-tuning, retrieval, or code-like prompts to reduce task-specific engineering. Other work concentrates on entity linking, ontology alignment, confidence estimation, knowledge fusion, graph validation, benchmarks, or evaluation rather than extraction alone.

The central finding of this review is that this history is not a simple sequence in which a new family makes all earlier families obsolete. Progress has changed where knowledge and constraints are placed. Early systems put them in hand-written patterns, parsers, dictionaries, and ontologies. Statistical and neural methods place more of them in learned parameters and training data. Generative and LLM methods also place them in prompts, examples, output grammars, retrieval systems, and post-processing. Modern systems frequently reintroduce explicit constraints because fluent generation is not the same as factual, canonical, or graph-valid generation. The most reliable designs in the acquired corpus are therefore combinations of methods rather than pure replacements.

This review answers eight questions:

1. Which methodology families are used to construct knowledge graphs from text?
2. Which stages of the Text-to-Knowledge-Graph process does each family address?
3. How has the field evolved from linguistic rules to transformers and LLMs?
4. How are systems evaluated?
5. What does each family do well?
6. What are its main weaknesses?
7. How can present systems be improved?
8. Which problems remain unresolved?

The contribution is a synthesis organized around methods, pipeline stages, and design trade-offs. Individual papers are used as evidence for broader findings; they are not presented as isolated summaries. Following the editorial principle that a review should make and support a clear argument, the discussion identifies the strongest recurring pattern in the evidence: flexible extraction is useful only when paired with mechanisms that control identity, structure, and factual quality.

## 2. Review basis and method

### 2.1 Evidence base

The project retained 115 candidate papers as relevant to transforming unstructured natural-language text into entities, relations, triples, schemas, or graphs. Complete sources were obtained and structured Paper Cards were accepted for 83 papers. Thirty-one retained papers still lacked a usable full text, and one acquired paper remained at Paper Card manual review. The synthesis therefore concerns the 83 successfully obtained and analyzed papers, not all 115 retained candidates and not the entire global literature.

Each accepted Paper Card records bibliographic metadata, the research problem, methodology, Text-to-KG stages, input and output, ontology or schema use, datasets, baselines, evaluation, reported results, strengths, weaknesses, author-stated limitations, possible improvements, unresolved problems, reproducibility information, and page-level evidence. Claims in this review were drawn from those cards. Source-reported results are distinguished from this review's interpretation.

### 2.2 Scope

Included work uses unstructured natural-language text as an input and produces information that directly constructs or populates a knowledge graph. A paper can cover one stage, such as relation extraction or entity linking, when its output is usable in a Text-to-KG process. The corpus also includes benchmarks and substantial surveys that help explain evaluation or historical development. Work concerned only with querying, embedding, completing, or reasoning over an already existing graph is outside the synthesis unless it also constructs graph content from text.

### 2.3 Synthesis procedure

The analysis uses twelve broad methodology families: rule-based NLP; linguistic and dependency-based extraction; classical machine learning; neural NLP pipelines; transformer encoders; joint entity-relation extraction; generative text-to-structure methods; zero-shot and few-shot LLM prompting; instruction-tuned or fine-tuned LLMs; ontology-constrained or schema-guided methods; hybrid NLP–LLM methods; and validation, correction, or knowledge-fusion methods. Papers may use several families, but the synthesis considers the main technical mechanism and important secondary mechanisms.

Results are compared only when tasks, datasets, metrics, and matching rules are compatible. A higher F1 score on one dataset does not establish superiority over a method evaluated elsewhere. This restriction is essential because the corpus spans sentence-level relation extraction, document-level OpenIE, entity linking, ontology population, RDF generation, graph fusion, domain applications, and human-reviewed end-to-end systems.

### 2.4 Limitations of the review

The evidence base is broad but incomplete. Full text was unavailable for 31 retained papers. Some accepted papers do not report a publication year or complete identifier in the acquired version. Methodology labels are interpretive categories used for synthesis rather than claims made uniformly by the original authors. The corpus contains many extraction benchmarks but fewer controlled evaluations of complete graph quality, graph maintenance, human use, or long-term deployment. Recent LLM papers are numerous, but their models, prompts, prices, and software environments change quickly. These limitations are treated as constraints on the conclusions rather than filled with outside assumptions.

## 3. What Text-to-Knowledge-Graph construction involves

A Text-to-KG system can be understood as a sequence of questions about the source and the emerging graph.

1. **Preprocessing:** What text should be analyzed, and how should it be divided into documents, paragraphs, sentences, or tokens?
2. **Entity and concept extraction:** Which spans name people, organizations, objects, processes, places, properties, or domain concepts?
3. **Typing:** Which class does each entity belong to?
4. **Coreference and resolution:** Do different mentions in the text refer to the same local entity?
5. **Entity linking:** Does the entity correspond to an existing record in a knowledge base?
6. **Relation, triple, or event extraction:** What statement connects the entities, and under what context?
7. **Ontology or schema alignment:** Does the statement use the graph's permitted classes, properties, and domain/range constraints?
8. **Graph construction:** How are extracted statements represented and stored?
9. **Fusion and deduplication:** Which repeated or conflicting entities and facts should be merged?
10. **Validation and correction:** Is the fact supported by the text, structurally valid, and consistent with the graph?
11. **Updating:** How is new knowledge added without damaging or duplicating the existing graph?

These stages need not be implemented as separate software modules. Joint and generative systems may solve several at once. The distinction nevertheless remains useful because an end-to-end output can still fail at a specific conceptual stage. A fluent triple can have the wrong entity boundary, a non-canonical relation, an unsupported object, or a duplicate entity.

The output also varies. Some studies produce relation instances or triples, while others produce RDF, named graphs, ontology elements, or an operational graph in Neo4j. A triple extraction score therefore measures only part of the quality of a deployed knowledge graph. A usable graph also needs stable identifiers, consistent semantics, provenance, deduplication, and support for later updates.

For readers new to the field, several recurring terms are useful. An **ontology** is an explicit description of the kinds of things and relations allowed in a domain. A **schema** is the structural contract that states how graph data should be represented. **RDF** is a standard way to express graph statements as subject–predicate–object triples. **OpenIE** extracts relation phrases without requiring a fixed relation list. **Entity linking** connects a name in text to a stable record, such as the record for a person in Wikidata. **Coreference resolution** determines that expressions such as “the company” and its earlier name refer to the same entity. **Canonicalization** replaces equivalent names or relations with one consistent form. These operations turn extracted words into graph content that can be compared, queried, and maintained.

## 4. Historical development

### 4.1 Linguistic knowledge and explicit rules

Early methods made their assumptions visible. Specia and Motta combined tokenization, part-of-speech tagging, shallow patterns, dependency relations, WordNet similarity, ontology classes, and word-sense disambiguation (Specia and Motta, 2006). The system could add semantic relations when extracted arguments and verbs matched ontology-compatible patterns. Its limitations were equally explicit: shallow patterns missed some structures, previously unseen relations were difficult, and entities outside the ontology could not be annotated. The paper presented examples but left quantitative evaluation to future work.

OpenIE broadened the goal from extracting a predefined relation inventory to finding relational phrases directly from text. ReVerb constrained verb-centered relation phrases with syntactic and lexical rules (Fader et al., 2011). On its Web-sentence evaluation, it obtained an area under the precision–recall curve 30% above WOEparse and more than twice that of WOEpos and TextRunner. It processed 100,000 sentences in 16 minutes, compared with 11 hours for WOEparse. The same study showed the trade-off behind its efficiency: the constraints missed non-contiguous relations, unusual word orders, and n-ary relations, while argument extraction remained the dominant error source.

Later linguistic pipelines combined OpenIE with entity linking, semantic-role labeling, and RDF construction. Martinez-Rodriguez et al. integrated three entity-linking services, ClausIE, semantic-role labeling, and rule-based fallbacks (Martinez-Rodriguez et al., 2018). On 100 IT-news sentences, the integrated entity-linking layer reached F1 0.8827, while ClausIE relation extraction reached F1 0.628. In a manual evaluation of 50 RDF events, complete-triple precision was 0.51; removing noun-phrase association reduced it to 0.26. The difference between strong component scores and lower full-triple quality illustrates why full graph construction cannot be judged from entity or relation extraction alone.

OPIEC demonstrated the scale possible with linguistic OpenIE (Gashteovski et al., 2019). Its Wikipedia pipeline produced 341 million triples, of which about 49% carried at least one semantic annotation. A manual sample found 71% correct extractions, and confidence correlated strongly with precision. Yet only 29.7% of linked triples had a hit in DBpedia or YAGO under the study's entity-pair criterion. The paper cautioned that such a hit does not prove relation equivalence or direction. Scale therefore amplified both coverage and the need for disambiguation, canonicalization, and validation.

### 4.2 Statistical learning and weak supervision

Classical machine learning reduced dependence on hand-written extraction rules but introduced dependence on labels, features, and knowledge-base coverage. Early entity-linking work learned how surface forms and context connect text mentions to Wikipedia entities (Milne and Witten, 2008). Entity-disambiguation systems for knowledge-base population combined retrieval, mention context, and ranking (Dredze et al., 2010). The survey by Shen et al. later organized entity linking into candidate generation, candidate ranking, and NIL prediction (Shen et al., n.d.). It concluded that no universal state of the art could be identified because systems differed in domains, data, and modules.

Weak and distant supervision used existing graph facts as approximate labels. MULTIR allowed several relations to hold for one entity pair and treated sentence-level relation labels as latent (Hoffmann et al., 2011). At its highest reported sentential-recall point, it achieved 72.4% precision, 51.9% recall, and F1 60.5%. A restricted variant that did not model overlapping relations increased precision by 12 points but reduced recall by 26 points and reduced F1 to 40.3%. The study also documented a central problem of distant supervision: Freebase incompleteness caused correct extractions to be counted as errors, while noisy relation matches could mislead training.

Universal schemas placed surface patterns and knowledge-base relations in a shared matrix (Riedel et al., 2013). Factorization and neighborhood information made it possible to rank unseen facts without forcing every textual expression into a predefined relation at extraction time. On the study's Freebase-relation task, the NFE model obtained weighted mean average precision 0.69, compared with 0.57 for the SU12 baseline. For surface-pattern prediction, however, the simpler factorization model was stronger than NFE. This result shows that typed knowledge-base relations and broad textual patterns do not always benefit from the same constraints.

Confidence estimation became a separate concern rather than an implicit classifier output. Li and Grishman combined six slot-filling pipelines and trained a maximum-entropy correctness model before weighted response voting (Li and Grishman, 2013). Knowledge Vault treated Web extraction and prior knowledge as uncertain evidence and performed probabilistic fusion at Web scale (Dong et al., 2014). These systems established an important design principle that remains relevant for LLM pipelines: extraction should preserve and combine evidence rather than immediately treating every output as a fact.

### 4.3 Neural pipelines, contextual encoders, and joint extraction

Neural methods learned representations that were difficult to specify as rules. Systems learned relation patterns from webpages (Xia et al., 2019), performed multi-task scientific information extraction (Luan et al., 2018), adapted a coarse-domain graph to a finer biomedical domain using iterative distant supervision (Cai et al., 2023), and constructed domain graphs with separate entity and relation models (Liu et al., 2022). These methods improved contextual modeling, but their performance remained tied to annotation quality, label balance, and domain match.

The domain-specific RoBERTa pipeline of Liu et al. illustrates both the capability and the risk of component-level evaluation (Liu et al., 2022). It reported named-entity F1 83.07%, relation-extraction F1 97.57%, and entity-alignment F1 99.81%. The paper also noted that some entity labels had F1 below 50%, the relation inventory was small, manual annotation was costly, and the limited number of relation types might make the relation result appear high. Aggregate scores can therefore conceal weak minority classes and narrow task definitions.

Joint extraction reduces error propagation by learning entities and relations together. Wang et al. represented entities and relations with a graph scheme (Wang et al., 2018), while Luan et al. jointly modeled entities, relations, and coreference for scientific knowledge graphs (Luan et al., 2018). Cascade tagging (Wei et al., 2020), encoder–decoder generation (Nayak and Ng, 2020), and OneRel's relation-specific token-pair classification (Shang et al., 2022) offered different solutions to overlapping triples and shared entities.

OneRel achieved exact-match F1 92.9 on NYT and 91.0 on WebNLG, and partial-match F1 92.8 and 94.3 on the corresponding relaxed settings (Shang et al., 2022). It obtained the best F1 in the study's four main comparisons and was strongest on 13 of 18 overlap and triple-count subsets. These results are strong within those datasets, but they should not be read as end-to-end graph quality: the benchmark assumes a fixed relation inventory and evaluates extracted triples rather than linking, fusion, or graph maintenance.

Document context became a further boundary. DocIE represented a source sentence together with surrounding sentences and used a BERT-based encoder–decoder (Dong et al., 2021). It reached F1 60.8 in healthcare patents and 56.9 in transportation patents, improvements of 1.0 and 0.8 points over the strongest sentence-level neural baseline reported in the study. The best context window differed by domain, and larger windows could add noise. The dataset also did not annotate coreference, showing that adding context does not automatically solve reference resolution.

### 4.4 Generative and LLM-based construction

Generative methods recast extraction as translation from text to a structured sequence. This makes one model capable of producing entities, relations, and graph structure, but the chosen serialization becomes part of the method. The autoregressive text-to-graph model ATG dynamically represented candidate spans and relation types, then generated nodes and edges under decoding constraints (Zaratiana et al., 2024). It reached relation-plus-entity F1 38.6 on SciERC, 66.2 on ACE05, and 78.5 on CoNLL04. Sorted graph linearization outperformed random ordering by 4.7, 4.9, and 1.0 points, respectively. Output order, therefore, was not a cosmetic choice; it changed what the model learned.

Semi-supervised generative OpenIE combined T5, pseudo-labels, a mean teacher, and a learned verification task (Zhou, 2023). On LSOIE-wiki, verification raised F1 from 68.2 for the supervised UIE-OIE model to 70.1. On CaRB, however, the no-verification semi-supervised variant scored 42.8, slightly above the 42.1 obtained with verification. The paper attributed difficulty partly to incompatible annotation specifications and noisy gold data. Verification itself must match the task's definition of a correct extraction.

LLMs further reduce the need for task-specific training. LlmRe used prompts to identify candidate entities and then ask whether schema-compatible triples were supported (Zhao et al., 2023). It achieved F1 0.854 on Movie and 0.813 on People, above the reported UIE and ChatIE comparisons, but scored 0.770 on Company, below UIE's 0.827. The variation across three fields cautions against treating zero-shot adaptability as uniform.

CodeKGC expressed schemas and outputs as Python-like code (Bi et al., 2024). With text-davinci-003, few-shot relation-strict F1 reached 64.2 on ADE, 49.6 on CoNLL04, and 24.7 on SciERC, improving over its matched vanilla prompts by 5.4, 6.4, and 5.9 points. Later evidence complicates the idea that programming-language prompts are inherently better. Gajo and Barrón-Cedeño evaluated 175 model and prompt settings and found that prompt language had limited influence after fine-tuning; natural-language prompts were slightly better overall (Gajo and Barrón-Cedeño, 2025). Their strongest adapted models reached micro-F1 0.812 on ADE, 0.705 on CoNLL04, and 0.396 on SciERC. Across tested settings, fine-tuned models averaged 0.719 versus 0.192 on ADE, 0.582 versus 0.087 on CoNLL04, and 0.305 versus 0.032 on SciERC. Under this limited-data setup, adaptation and output-format supervision mattered more than code specialization or model size.

Fine-tuning can outperform a large proprietary model on a matched task without guaranteeing transfer. Zhang et al. used LoRA and augmented instructions, explanations, reverse generation, and reflection examples (Zhang et al., 2024b). On one WebNLG setting, Orca-mini-3-7B achieved strict F1 0.717 compared with 0.645 for the listed GPT-4 baseline, while Vicuna-33B reached 0.724. The same models transferred poorly to DocRED: strict F1 was 0.024 for Orca-mini-3 and 0.002 for Llama-2-13B. The paper also observed hallucinated facts and output loops. Task adaptation can be strong and still be brittle outside the training representation.

The field is increasingly moving from direct generation to multi-stage LLM pipelines. iText2KG distilled documents into structured blocks, incrementally resolved entities and relations, and loaded them into Neo4j (Lairgi et al., 2024). Local entity context produced higher relevant-triple precision than global context in computer science (0.94 versus 0.83) and music (0.90 versus 0.81), while global context could recover implied relations at the cost of irrelevant edges. Extract–Define–Canonicalize separated open extraction, relation definition, and schema canonicalization (Zhang and Soh, 2024a). On WebNLG, its GPT-4 EDC+R variant reached exact F1 0.800, compared with 0.723 for REGEN; on REBEL it reached 0.574, compared with 0.364 for GenIE. The authors nevertheless reported many LLM calls, hallucination and privacy risks, and unresolved entity duplication.

KGGen combined LLM entity and triple extraction with explicit entity and relation resolution (Mo et al., 2025). On MINE-1, it recovered 66.07% of facts, compared with 47.80% for GraphRAG and 29.84% for OpenIE in the same experiment. Human assessment found 98 of 100 KGGen triples valid, compared with 55 of 100 OpenIE triples and none of the 100 GraphRAG triples under the paper's validity criteria. On a one-million-character corpus, KGGen took 551 seconds and cost USD 0.84, while GraphRAG extraction took 2,319 seconds. These results support the value of resolution, but the paper also warned that resolution may merge distinct items or miss duplicates, and its benchmarks did not reach Web scale.

The historical pattern is therefore cumulative. LLMs introduce flexible semantic generation, but their successful systems use the same kinds of controls that earlier work identified: candidate restriction, schemas, confidence, verification, canonicalization, provenance, and domain knowledge.

## 5. Comparative synthesis of methodology families

Table 1 summarizes the main role and trade-off of each family. Coverage labels describe representation in the acquired corpus, not the size of the field as a whole.

| Methodology family | Main idea | Main strengths | Main limitations | Corpus coverage |
| --- | --- | --- | --- | --- |
| Rule-based NLP | Encode extraction decisions as patterns, dictionaries, and deterministic logic | Transparent, controllable, low data requirement | Narrow coverage; expensive maintenance; brittle language variation | Well covered |
| Linguistic/dependency extraction | Use grammatical structure, semantic roles, and OpenIE | Interpretable relation structure; can work without a fixed schema | Parser and argument errors; heterogeneous relation phrases | Well covered |
| Classical machine learning | Learn classifiers or rankings from engineered features, direct labels, or weak labels | Efficient; confidence can be calibrated; mature evaluation | Feature and label dependence; distant-supervision noise | Well covered |
| Neural NLP pipelines | Learn contextual features for separate entity, relation, or classification modules | Better generalization than fixed patterns; modular deployment | Error propagation; annotation and domain dependence | Adequately covered |
| Transformer encoders | Use pretrained contextual representations | Strong context modeling; effective transfer and multilingual use | Compute cost; long-document and rare-class limits | Adequately covered |
| Joint entity-relation extraction | Predict entities and relations together | Reduces pipeline separation; handles overlapping triples | Usually fixed schemas and benchmark settings; costly pair/span search | Well covered |
| Generative text-to-structure | Generate triples, trees, code, RDF, or graphs as sequences | Unified output; flexible structures; direct graph serialization | Invalid output, ordering bias, hallucination, decoding cost | Well covered |
| Zero-/few-shot LLM prompting | Specify the task with instructions, schemas, and examples | Low task-specific training; rapid adaptation | Prompt sensitivity, cost, privacy, unstable formatting | Well covered |
| Fine-tuned/instruction-tuned LLMs | Adapt an LLM to extraction and output conventions | Strong matched-task performance; can use smaller models | Data quality and transfer limits; training cost; hallucination remains | Adequately covered but less mature |
| Ontology/schema-guided methods | Restrict or align extraction to known classes and relations | Interoperable and structurally valid output | Ontology coverage and token limits; rejects unknown knowledge | Adequately covered |
| Hybrid NLP–LLM methods | Combine deterministic/NLP modules with LLM extraction or validation | Balances flexibility and control | More components, latency, integration errors, external-service risk | Adequately covered |
| Validation/correction/fusion | Score, link, merge, check, or correct extracted knowledge | Converts noisy extractions into more usable graph content | Depends on KB completeness and thresholds; can lower recall | Well covered |

### 5.1 Rule-based NLP

Rule-based systems remain valuable when the language, relation inventory, or compliance requirements are stable. ReVerb shows that carefully designed syntactic and lexical constraints can be both fast and competitive (Fader et al., 2011). DBpedia Spotlight and related linking work demonstrate the continuing importance of dictionaries, surface forms, and deterministic candidate control (Mendes et al., n.d.). Pattern systems also appear in domain settings, such as lexical extraction from Spanish text (Rios-Alvarado et al., 2022), biomedical graph generation (Rossanez et al., 2020), and a knowledge-graph contest pipeline (Stewart et al., 2019).

Their central advantage is inspectability. A reviewer can see why a pattern fired and can change it without retraining a model. Their central weakness is coverage. Language expresses the same meaning in many forms; a rule precise enough to avoid false positives may miss paraphrases, long-distance dependencies, implicit arguments, and domain-specific terminology. Rules are therefore strongest as constraints or support modules rather than as the only source of semantics. This continued role is visible in recent systems: LLM-TIKG still uses regular expressions for indicators of compromise and rules for filtering and alias merging (Hua et al., 2023), while G-T2KG uses noun-phrase and dependency rules around OpenIE and GPT-4 (Kabal et al., 2024).

### 5.2 Linguistic and dependency-based extraction

Linguistic systems represent how words relate grammatically. Dependencies, constituency trees, semantic roles, and clause structures help separate subjects, predicates, and objects. Their output can remain open-domain, as in OpenIE, or be grounded to a knowledge base. The family is especially useful when interpretability and broad relation discovery are more important than a closed label inventory.

Open extraction and grounding solve different problems. Yu et al. used dependency graphs, random-walk distances, entity types, and relation embeddings to extract and ground binary relations (Yu et al., 2017). On KBP2013, their joint relation representation obtained F1 28.9%, above the reported UW OpenIE baseline's 20.8%, although the baseline had much higher precision and much lower recall. The result is not simply a contest between methods; it shows two operating points—conservative extraction and broader grounded coverage.

The main recurring failure is incorrect or incomplete arguments. ReVerb identifies argument extraction as its largest error source (Fader et al., 2011). OPIEC reports under-specified triples because it does not resolve coreference (Gashteovski et al., 2019). Document-level methods improve access to context but can add noise (Dong et al., 2021). Linguistic extraction therefore benefits from explicit coreference, entity linking, and confidence layers.

### 5.3 Classical machine learning

Classical machine learning brought trainable ranking and classification to relation extraction, linking, and fusion. It remains conceptually important because many current systems still depend on the same design elements: candidate generation, feature representation, negative sampling, confidence calibration, and aggregation.

Distant supervision made large training sets possible by aligning known graph facts with sentences, but it made labels uncertain. MULTIR addressed overlapping relations with latent sentence labels (Hoffmann et al., 2011). Universal schemas jointly represented surface patterns and graph relations (Riedel et al., 2013). Distantly supervised Web extraction (Augenstein et al., 2016) and semi-automatic ontology population (Elkhammash and Abdessalem, 2019) extended the same general strategy into other settings. These papers show that more training data do not automatically mean better supervision. Missing graph facts, wrong alignments, skewed relations, and incomplete ontologies directly affect both learning and evaluation.

Classical methods are often efficient and easier to diagnose than large neural systems. MULTIR reported roughly one minute of training and under one second of testing in its setting, compared with hours for its cited alternatives (Hoffmann et al., 2011). Their weakness is that hand-designed features and task-specific models can struggle with new domains and complex context. Their continued value lies in calibrated decision-making and fusion, where transparent scores and thresholds remain useful.

### 5.4 Neural pipelines and transformer encoders

Neural pipelines usually maintain separate stages but replace manually designed feature sets with learned representations. This makes them practical for domain-specific systems because individual modules can be trained, evaluated, and replaced. MWO2KG, for example, uses a character-aware neural NER model for noisy maintenance text and a separate failure-mode classifier before deterministic triple generation and Neo4j storage (Stewart et al., 2024). Its NER reached micro-F1 0.828, but failure-mode classification reached only 0.597, with several minority classes at zero F1. The graph therefore inherits a strong entity component and a much weaker classification component.

Iterative neural adaptation can expand a graph with less direct annotation. KGDA bootstraps NER and relation models from a coarse biomedical graph and adds high-confidence predictions over successive corpus partitions (Cai et al., 2023). Its held-out relation models reported F1 around 0.97, but manual precision for extracted target-domain entities and relations was substantially lower. The authors called for denoising and more human evaluation of relevance. This difference again separates benchmark classification from trustworthy graph population.

Transformers improve contextual representation and multilingual transfer. DocIE adds document context (Dong et al., 2021). RoBERTa-based domain construction combines NER, relation extraction, and alignment (Liu et al., 2022). XLM-RoBERTa with synthetic Kazakh data reached F1 90.73%, compared with 82.6% for BERT with synthetic data and 74.9% without it in the reported comparison (Bektemyssova et al., 2026). Yet the study did not fully report corpus provenance or splits, and synthetic data lacked dialectal coverage. High scores require enough information about data construction to be interpretable.

### 5.5 Joint entity-relation extraction

Joint methods recognize that entity and relation predictions depend on one another. They use graph labeling, cascade tagging, table filling, token-pair classification, or autoregressive decoding to avoid a strict entity-first pipeline (Wang et al., 2018; Wei et al., 2020; Shang et al., 2022; Nayak and Ng, 2020; Baek et al., 2025). They are particularly useful for overlapping triples, where one entity appears in several relations or two entities have several relations.

The family has strong benchmark evidence. OneRel reports F1 above 90 on the NYT and WebNLG variants in its study (Shang et al., 2022). The encoder–decoder models of Nayak and Ng reached F1 0.682 on NYT29 and 0.817 on NYT24 for word decoding, while pointer decoding was more than twice as fast and used one-third of the GPU memory (Nayak and Ng, 2020). The faster model made more entity–relation pairing errors. Accuracy and efficiency therefore depend on how the output structure is decoded.

Joint extraction is not the same as full Text-to-KG construction. Most joint benchmarks assume already segmented examples, known entity and relation types, and exact or partial tuple matching. They generally do not evaluate identity across documents, ontology evolution, conflicting facts, provenance, or graph updates. Joint models solve an important internal dependency but still require downstream graph engineering.

### 5.6 Generative text-to-structure methods

Generative methods represent structure as an output language. The output may be a tuple sequence, tree, graph linearization, Python-like code, JSON, or RDF. This approach unifies several predictions and can naturally emit a variable number of facts. The output language, ordering, and decoding constraints become critical sources of inductive bias.

ATG's sorted graph order clearly outperformed random order (Zaratiana et al., 2024). Semi-supervised UIE-OIE found that a tree representation outperformed a tuple sequence on both LSOIE-wiki and CaRB (Zhou, 2023). The extensive RDF benchmark by Ringwald et al. showed that syntax affected validity, accuracy, output length, time, and emissions (Ringwald et al., 2026). T5 with JSON-LD reached macro F1+ 95.63 with fully valid outputs in that restricted datatype-property task. A BART factorized Turtle Light configuration reached 94.54 while requiring about one-fifth of the training time and far lower reported emissions. Grammar-constrained decoding, however, was roughly six times slower and still produced incomplete or invalid output in the tested implementation.

These findings show that serialization is not merely presentation. A shorter, regular representation reduces the burden on the decoder. Conversely, a standard format such as RDF/XML or JSON-LD may be harder for a model even when it is more interoperable after generation. Practical systems may therefore generate a simpler controlled representation and convert it deterministically into standards-compliant RDF.

### 5.7 Zero-shot and few-shot LLM prompting

Prompted LLM systems can define a task through natural-language instructions, a schema, and a small number of examples. They are attractive when labeled data are scarce or the ontology changes frequently. The corpus includes direct triple extraction (Zhao et al., 2023), code-style prompting (Bi et al., 2024), incremental construction (Lairgi et al., 2024), ontology-verified extraction (Chepurova et al., 2024), schema canonicalization (Zhang and Soh, 2024a), materials-science applications (Kaverinskiy, 2025; Bai et al., 2025), and broader reviews (Bottino and Alcázar, 2025; Zhu et al., 2024).

The evidence does not support a claim that prompting alone solves Text-to-KG construction. In the ontology-verification pipeline of Chepurova et al., the full prompted system reached F1 0.55 on SynthIE-text-small, while the fine-tuned SynthIE T5-large baseline reached 0.88 (Chepurova et al., 2024). On a human-evaluated natural-language corpus, however, the prompted method had precision 0.74 versus 0.55 for SynthIE, and the systems had low overlap in their correct triples. Narrow automatic benchmarks and natural text can favor different methods.

Prompt design matters. Removing in-context examples from the same pipeline reduced F1 from 0.55 to 0.16 (Chepurova et al., 2024). CodeKGC benefited from schema-aware code and rationales in several few-shot settings (Bi et al., 2024), while rationale training reduced performance for almost every fine-tuned model tested by Gajo and Barrón-Cedeño (Gajo and Barrón-Cedeño, 2025). A technique that helps one model or supervision regime may burden another. Prompt components should therefore be treated as experimental factors, not universal recipes.

### 5.8 Instruction-tuned and fine-tuned LLM methods

Fine-tuning teaches both the extraction task and its output convention. It can improve small models substantially and offers local deployment possibilities. The corpus now includes general triple extraction (Zhang et al., 2024b; Gajo and Barrón-Cedeño, 2025) and specialized threat-intelligence systems (Hua et al., 2023; Yang et al., 2026).

Data quality is more important than raw volume in several studies. LLM-TIKG found that a manually corrected 1,600-example set produced much better NER precision than a 15,000-example coarsely filtered set (Hua et al., 2023). Its 1,600-example LLaMA model reached NER F1 85.89, and its TTP technique classifier reached F1 98.21. The deployed graph contained 50,745 entities and 64,948 relations from 9,681 threat reports. The authors nevertheless reported long-text omissions, ambiguous entity boundaries, scarce authoritative annotations, and errors in GPT-generated labels.

CTI-Thinker combined LoRA adaptation, retrieved demonstrations, semantic alignment, and graph retrieval (Yang et al., 2026). It reported entity F1 0.8385, relation F1 0.7689, and best thresholded entity-alignment F1 0.8383. On 30 reasoning queries, accuracy was 0.635, compared with 0.615 for LLM-RAG and 0.223 for a native LLM. The remaining limitations—semantic drift, redundant triples, unstable multi-turn reasoning, and weak reasoning-path consistency—show that strong extraction does not guarantee dependable downstream reasoning.

Fine-tuning is therefore not a final stage in the field's evolution. It is one control mechanism among several. Its value depends on task-matched data, careful annotation, appropriate output representation, and validation after generation.

### 5.9 Ontology-constrained and schema-guided methods

Ontologies define classes, properties, and constraints. They improve interoperability and can prevent structurally invalid facts. Earlier systems used ontology compatibility to select relations (Specia and Motta, 2006), entity linkers used semantic knowledge to rank candidates (Shen et al., 2012), and biomedical systems linked mentions to domain resources (Zheng et al., 2015). Modern LLM systems place a verbalized ontology directly in the prompt or retrieve candidate schema elements.

Text2KGBench measures not only exact triples but also ontology compliance and hallucinated subjects, relations, and objects (Mihindukulasooriya et al., 2023). Vicuna-13B reached F1 0.35 on all Wikidata-TekGen cases and 0.30 on DBpedia-WebNLG in the reported setup; ontology compliance was higher than exact extraction quality. Performance varied considerably across ontologies, and the benchmark used small ontologies because of context-window limits. This distinction is important: following a schema does not prove that the extracted fact is correct.

Schema constraints can also remove true facts when the graph is incomplete. In the news-stream pipeline, RDF validation improved precision from 54.5 to 70.1 and F1 from 66.6 to 75.5, while recall fell from 85.5 to 81.7 (Fernández Cañellas, 2023). The paper noted that valid triples can be rejected when required entity types are missing. Ontologies improve control, but their incompleteness becomes a source of false negatives.

### 5.10 Hybrid NLP–LLM methods

Hybrid systems allocate work according to the strengths of different components. Deterministic code handles parsing, formats, regular patterns, constraints, and storage. Encoders handle contextual classification. LLMs handle flexible extraction, interpretation, or validation. Humans address ambiguous or high-risk cases.

G-T2KG is a clear example (Kabal et al., 2024). OpenIE6 and a hypernym extractor generate candidates; syntactic or GPT-based noun-phrase cleaning improves them; GPT-4 validates each triple against its sentence; and mapping reduces label heterogeneity. On the computer-science corpus, the dependency-based option moved from precision 58.50 and recall 47.77 without validation to precision 72.07 and recall 44.44 with validation. Across its experiments, validation increased precision by 14–25% while reducing recall by at most 3.3%. The cost of this control is an external dependency unsuitable for confidential text.

Domain systems add further engineering. The cultural-heritage ATR4CH pipeline combines LLM metadata extraction, GLiNER entity detection, LLM classification and coreference, rule-based Wikidata linking, and RDF-star mapping (Schimmenti et al., 2025). Metadata extraction reached F1 0.991 with Claude, but entity/cognizer recognition was lower, with the best reported F1 0.803 for GPT-4o-mini. Generated graphs contained substantially more triples than the gold graph, and human post-processing remained necessary. The weak link in a hybrid pipeline may therefore be a specific semantic classification stage rather than the LLM in general.

In scientific question–answer generation, a KG-based pipeline selected salient triples before asking an LLM to form questions and answers (Azarbonyad et al., 2025). Subject-matter experts rated it above a context-only pipeline across relevance, specificity, clarity, factuality, and completeness in the two tested domains. The result supports graph-mediated content selection, but evaluation covered only 200 question–answer pairs. Hybrid design can improve downstream use while still requiring broader validation.

### 5.11 Validation, correction, and knowledge fusion

Validation and fusion are important components when the target application requires durable, trustworthy graph knowledge. They determine whether extracted strings become durable graph knowledge. The family includes confidence estimation (Li and Grishman, 2013), probabilistic fusion (Dong et al., 2014), integration of entity extraction and linking systems (Hernandez et al., 2021), simultaneous entity and relation grounding (Lin et al., 2020), ontology growth (Suchanek, 2009), and surveys/resources for entity linking and KB population (Ji and Grishman, 2011; Shen et al., n.d.; Getman et al., n.d.).

FEEL integrates several entity extraction and linking services, removes duplicates and overlaps, and applies voting or frequency filters (Hernandez et al., 2021). A stricter configuration generally increased precision and reduced recall. Its results also differed across seven datasets, and the authors warned that changing knowledge-base identifiers could make benchmark annotations appear wrong. Fusion quality depends on the sources, their correlation, and the freshness of the evaluation data.

KBPearl constructs a graph containing noun phrases, relation phrases, candidate entities, and candidate predicates, then jointly selects a dense assignment (Lin et al., 2020). Its nearest-neighbor variant reached entity-linking F1 0.575 on NYT2018 and 0.421 on T-REx, and its MinIE pipeline reached F1 0.561 on QALD-7-Wiki. The method remained more stable on long documents than a similarity-only variant. Joint grounding can use global document evidence, but performance remains far below the extraction scores often reported on closed relation benchmarks.

Evaluation methodology is itself part of trustworthy validation. Chaganty et al. proposed importance-sampled, on-demand evaluation and reported an order-of-magnitude reduction in required labels compared with fixed sampling (Chaganty et al., n.d.). The method depends on assumptions about sampling true instances and uses heuristic mixing weights, but it addresses a real problem: static test sets can become biased against new systems.

## 6. Coverage of the Text-to-KG pipeline

### 6.1 Preprocessing and document selection

Preprocessing ranges from tokenization to full document acquisition and filtering. Traditional systems expose each operation, while LLM papers sometimes compress preprocessing into chunking. The choice still matters. News-stream construction must crawl, deduplicate, cluster, and track articles before extraction (Fernández Cañellas, 2023). Scientific systems must select relevant paragraphs (Azarbonyad et al., 2025). Long documents must be chunked without losing cross-sentence evidence (Lairgi et al., 2024; Hua et al., 2023).

The literature does not establish one best chunk size or context strategy. DocIE found domain-specific optimal context windows (Dong et al., 2021). iText2KG found local context more precise and global context richer but noisier (Lairgi et al., 2024). These studies support adaptive context selection rather than sending every available token to the model.

### 6.2 Entity extraction and typing

Entity extraction is among the best-covered stages, but aggregate F1 can hide operational weaknesses. MWO2KG's per-class F1 ranged from 0.640 to 1.000 (Stewart et al., 2024). The military-domain RoBERTa model had overall F1 83.07% but several labels near or below 50% (Liu et al., 2022). LLM-TIKG improved some threat-intelligence entities yet still missed entities in long text and tables (Hua et al., 2023).

The evidence supports three practices: report per-class results; preserve uncertainty at ambiguous boundaries; and evaluate the effect of entity errors on triples and graph use. Entity typing should not be assumed from entity recognition. G-T2KG explicitly identifies typing as future work (Kabal et al., 2024), while joint systems often predict types only from a fixed benchmark inventory.

### 6.3 Relation, triple, and joint extraction

Relation and triple extraction have the densest benchmark evidence. Methods range from patterns and dependency paths to classifiers, table filling, span graphs, and autoregressive decoders. Within matched tasks, joint and adapted generative systems often report strong results. Yet relation extraction is particularly sensitive to matching rules. A partial match may require only the relation and final words of entity spans; exact matching requires complete spans; strict matching may also require correct types. Results must always be read with the stated criterion.

OpenIE and schema-based extraction also optimize different targets. OpenIE values recall and discovery of relations not already named in a schema. Schema-based systems value canonical, interoperable predicates. A broad OpenIE triple such as a surface verb phrase may be useful evidence but is not immediately equivalent to a stable graph property. Alignment and canonicalization form the bridge (Gashteovski et al., 2020; Zhang and Soh, 2024a).

### 6.4 Coreference, entity resolution, and linking

The same entity may be named by a full name, abbreviation, pronoun, alias, or description. Coreference resolves mentions within text; entity resolution merges duplicate graph nodes; entity linking connects a mention to an external identifier. These are different tasks, though papers sometimes combine them.

The entity-linking literature shows that candidate generation can cap the performance of all later ranking (Shen et al., n.d.). Context, popularity, type compatibility, and coherence provide complementary evidence. Domain resources matter: biomedical linking must handle specialist terminology (Zheng et al., 2015), while news streams must create provisional entities for emerging people not yet in the graph (Fernández Cañellas, 2023).

LLM generation does not remove the identity problem. iText2KG uses incremental similarity matching (Lairgi et al., 2024). KGGen applies embeddings, lexical retrieval, clustering, and LLM decisions (Mo et al., 2025). EDC addresses relation canonicalization but leaves entity duplication unresolved (Zhang and Soh, 2024a). Entity identity remains a separate engineering and evaluation requirement.

### 6.5 Ontology alignment and graph construction

An ontology provides shared meaning and constraints; graph construction provides a persistent representation. Ontology-first methods improve consistency but may miss novel relations. Schema induction preserves novelty but can create synonymous or overly specific relation types. EDC explicitly supports both target alignment and self-canonicalization (Zhang and Soh, 2024a). KGGen measures relation reuse as the corpus grows (Mo et al., 2025). These are useful moves beyond isolated triple accuracy because they consider whether the graph's vocabulary remains coherent.

Graph serialization also deserves evaluation. The RDF benchmark evaluates parseability, subject correctness, SHACL validity, extraction metrics, time, and emissions (Ringwald et al., 2026). The OpenIE-to-RDF pipeline evaluates complete event representations and provenance (Martinez-Rodriguez et al., 2018). The evidence supports generating a constrained intermediate form, validating it deterministically, and only then loading it into a graph store.

### 6.6 Validation, fusion, and updating

Validation may compare a triple with its source sentence, enforce ontology constraints, combine redundant evidence, estimate confidence, or ask a human. Each method has a different failure mode. Source-sentence LLM validation can improve precision but expose confidential text (Kabal et al., 2024). Schema validation can reject true facts when type data are missing (Fernández Cañellas, 2023). Confidence learned from an incomplete knowledge base can inherit its omissions (Li and Grishman, 2013). Human review is expensive but remains necessary for ambiguous, high-impact, or emerging entities.

Updating is less developed than extraction. The traditional Chinese medicine system grows a graph in response to searches and reported 107 cumulative entities after ten retrievals (Xu et al., 2022), but the authors acknowledged weaknesses in merging and left implicit relation discovery for future work. News-stream construction supports continuous events and emerging entities (Fernández Cañellas, 2023). Most other systems evaluate a static dataset. The field can build graphs more readily than it can maintain their identity, provenance, conflicts, and temporal meaning over time.

## 7. How Text-to-KG systems are evaluated

### 7.1 Extraction metrics

Precision, recall, and F1 are the most common measures. Precision asks what proportion of predicted items are correct. Recall asks what proportion of reference items were found. F1 is their harmonic mean. These measures are useful only after defining an item and a match. An item can be an entity span, a linked entity, a relation label, a complete triple, or a graph statement. A relaxed match can credit partial entity boundaries, while a strict match may require exact boundaries, types, and relation direction.

The importance of the definition is visible within individual papers. OneRel reports separate partial- and exact-match results (Shang et al., 2022). The fine-tuning study reports type, partial, exact, and strict triple scores (Zhang et al., 2024b). The RDF benchmark adds parseability, correct subject, SHACL validity, and datatype-property measures (Ringwald et al., 2026). No single number can replace this detail.

Table 2 gives representative measurements from compatible comparisons within individual studies. It does not rank the studies against one another.

| Study and task | Compared methods or variants | Reported result | Supported interpretation |
| --- | --- | --- | --- |
| ReVerb, Web OpenIE (Fader et al., 2011) | ReVerb vs WOEparse, WOEpos, TextRunner | ReVerb AUC was 30% above WOEparse and more than twice WOEpos/TextRunner; 100,000 sentences took 16 minutes vs 11 hours for WOEparse | Linguistic constraints improved the study's precision–recall and speed balance |
| MULTIR, distant supervision (Hoffmann et al., 2011) | Full overlap model vs restricted training | Full model F1 60.5%; restricted variant 40.3% | Modeling multiple relations preserved recall under weak labels |
| Universal schema (Riedel et al., 2013) | MI09, YA11, SU12, N, F, NF, NFE | NFE weighted MAP 0.69 vs SU12 0.57; NF had highest unweighted MAP 0.66 | Factorization and neighborhood evidence were complementary, but weighting changed the preferred model |
| OpenIE-to-RDF pipeline (Martinez-Rodriguez et al., 2018) | Full pipeline vs no noun-phrase association | Complete-triple precision 0.51 vs 0.26 | A supporting alignment stage materially affected full output quality |
| OneRel (Shang et al., 2022) | OneRel and joint-extraction baselines | Exact F1 92.9 on NYT and 91.0 on WebNLG | One-step token-pair classification was strong on fixed-schema sentence benchmarks |
| DocIE (Dong et al., 2021) | Document context vs strongest sentence baseline | F1 60.8 vs 59.8 in healthcare; 56.9 vs 56.1 in transportation | Context helped modestly and required domain-specific window selection |
| Semi-supervised UIE-OIE (Zhou, 2023) | With vs without learned verification | LSOIE-wiki F1 70.1 vs 69.4; CaRB 42.1 vs 42.8 | Verification helped under one annotation scheme but not another |
| Ontology-verified prompting (Chepurova et al., 2024) | Full prompted pipeline vs fine-tuned SynthIE | F1 0.55 vs 0.88 on SynthIE-text-small; human-corpus precision 0.74 vs 0.55 | Fine-tuning led on the matched benchmark, while prompting was more precise on the tested natural text |
| G-T2KG (Kabal et al., 2024) | Dependency cleaning with vs without GPT-4 validation | Precision 72.07% vs 58.50%; recall 44.44% vs 47.77% | Validation traded a small amount of recall for substantially higher precision |
| KGGen (Mo et al., 2025) | KGGen, GraphRAG, OpenIE on MINE-1 | Fact recovery 66.07%, 47.80%, 29.84%; manually valid triples 98/100, 0/100, 55/100 | LLM extraction plus resolution improved fact recovery and output validity in the study |
| Natural vs code prompts (Gajo and Barrón-Cedeño, 2025) | Adapted vs base models | Mean micro-F1 0.719 vs 0.192 on ADE; 0.582 vs 0.087 on CoNLL04; 0.305 vs 0.032 on SciERC | Task adaptation mattered more than prompt language in the tested setup |
| News-stream RDF validation (Fernández Cañellas, 2023) | Before vs after validation | Precision 54.5 to 70.1; recall 85.5 to 81.7; F1 66.6 to 75.5 | Ontology and redundancy checks improved reliability with a recall cost |

### 7.2 Component versus end-to-end evaluation

Component evaluation helps locate errors. Separate NER, relation, linking, and alignment scores reveal which module limits a pipeline. End-to-end evaluation asks whether the final graph is usable. Both are necessary.

Several papers report strong components without a controlled complete-graph score. MWO2KG evaluates NER and failure-mode classification but not triple generation (Stewart et al., 2024). The domain RoBERTa study reports high relation and alignment results but notes a narrow relation inventory (Liu et al., 2022). Joint extraction benchmarks measure tuples but not cross-document identity (Shang et al., 2022). Conversely, human evaluation of complete RDF events exposes compound errors that component metrics can hide (Martinez-Rodriguez et al., 2018).

A minimum evaluation should therefore report: entity quality; relation or triple quality under a stated matching rule; identity/linking quality when applicable; structural validity; duplicate or canonicalization behavior; and at least one end-to-end measure tied to the intended use.

### 7.3 Graph-level quality

Graph-level evaluation is less standardized. The acquired studies use several proxies:

- RDF parsing and SHACL conformance (Ringwald et al., 2026);
- ontology compliance and hallucinated schema elements (Mihindukulasooriya et al., 2023);
- entity/relation reuse and duplicate reduction (Mo et al., 2025);
- confidence or probabilistic fusion (Dong et al., 2014; Li and Grishman, 2013);
- manual correctness of ingested triples (Fernández Cañellas, 2023);
- competency questions and expert assessment (Schimmenti et al., 2025);
- downstream question–answer quality (Azarbonyad et al., 2025);
- reasoning accuracy over graph-retrieved evidence (Yang et al., 2026).

These measures capture different properties. Structural validity proves that a graph follows a format or schema, not that its facts are true. Source support proves that a sentence supports a fact, not that the entity is linked correctly. Downstream performance can improve even when some graph errors remain. A sound evaluation should keep these dimensions separate.

### 7.4 Human evaluation

Human evaluation is necessary where gold graphs are incomplete, relation boundaries are open, or quality includes usefulness and interpretation. It is also vulnerable to small samples and unclear instructions. OPIEC manually judged 500 triples (Gashteovski et al., 2019). The OpenIE-to-RDF study used 50 RDF events and reported inter-rater agreement by component (Martinez-Rodriguez et al., 2018). ATR4CH included expert evaluation of scholarly debates (Schimmenti et al., 2025). The scientific QA study used subject-matter experts on 200 question–answer pairs (Azarbonyad et al., 2025).

Good practice is to report the sampling procedure, sample size, evaluator expertise, instructions, agreement, and whether annotators saw system identity. Several studies report only part of this information. Human assessment should not be treated as self-explanatory.

### 7.5 Efficiency, cost, and reproducibility

Runtime and cost matter because Text-to-KG construction can involve millions of sentences or repeated LLM calls. ReVerb compared minutes with hours (Fader et al., 2011). Pointer decoding reduced time and GPU memory relative to word decoding (Nayak and Ng, 2020). The RDF benchmark reported training time and carbon emissions (Ringwald et al., 2026). KGGen reported wall time and API cost (Mo et al., 2025). EDC estimated USD 0.009 per example for its GPT-3.5 components (Zhang and Soh, 2024a).

These measurements are useful but not directly comparable across hardware, time, providers, and workloads. Reproducibility also varies. Some papers release code, datasets, or gold standards; others report no availability. For proprietary LLMs, a prompt and model name may still be insufficient because the hosted model can change. Future work should preserve prompts, decoding parameters, model versions, schemas, preprocessing rules, raw and canonical outputs, and evaluation code.

## 8. Cross-family findings

### 8.1 The main trade-off is flexibility versus control

Rules and ontologies provide control but limit what can be discovered. OpenIE and LLMs provide flexibility but create more varied and potentially unsupported output. The literature repeatedly adds a control layer after broad extraction: confidence estimation (Li and Grishman, 2013), probabilistic fusion (Dong et al., 2014), ontology checks (Chepurova et al., 2024; Fernández Cañellas, 2023), LLM source validation (Kabal et al., 2024), canonicalization (Zhang and Soh, 2024a), or entity/relation resolution (Mo et al., 2025).

This pattern suggests that extraction and acceptance should be separate decisions. A system can generate candidates broadly, preserve their provenance and confidence, and admit them to the graph only after checks appropriate to the domain. High-risk applications may require a human decision; lower-risk settings may use calibrated thresholds.

### 8.2 End-to-end models reduce interfaces but not responsibilities

Joint and generative models remove explicit boundaries between entity and relation modules. This can reduce pipeline error propagation and represent overlapping triples. It does not remove the need to evaluate entity identity, relation semantics, schema validity, duplication, and provenance. These responsibilities remain even when hidden inside one model.

An end-to-end label should therefore describe architecture, not completeness. ATG generates graph structures (Zaratiana et al., 2024), but its benchmark does not evaluate cross-document graph fusion. Seq2KG jointly extracts triples and types (Stewart and Liu, n.d.), but a deployed graph may still need linking and validation. The right question is not whether a system is end to end, but which responsibilities its end actually includes.

### 8.3 More context is not always better

Cross-sentence context can resolve omitted arguments and references, but irrelevant context can distract extraction. DocIE found optimal windows that differed between two patent domains (Dong et al., 2021). iText2KG found local context more precise than global context (Lairgi et al., 2024). Threat-intelligence systems report omissions in long text (Hua et al., 2023). Context selection should therefore be retrieval- or structure-aware, with evaluation over long documents rather than assumed benefit from larger windows.

### 8.4 Data quality often dominates model size

MULTIR shows how incomplete knowledge bases distort weak labels and evaluation (Hoffmann et al., 2011). The military RoBERTa model shows how label imbalance weakens minority entities (Liu et al., 2022). Semi-supervised OpenIE identifies annotation noise as its largest sampled discrepancy (Zhou, 2023). LLM-TIKG reports that 1,600 corrected examples outperform 15,000 coarsely filtered examples for NER precision (Hua et al., 2023). Gajo and Barrón-Cedeño find that small adapted models can outperform a 70B model under a limited training regime (Gajo and Barrón-Cedeño, 2025).

The consistent lesson is that adding model capacity cannot repair an unclear task definition, noisy labels, inconsistent serialization, or missing graph facts. Corpus design and annotation policy are core methodology, not supporting details.

### 8.5 Validation improves precision but can hide incompleteness

Validation frequently raises precision and lowers recall (Kabal et al., 2024; Fernández Cañellas, 2023). This is often appropriate for graph population because an incorrect durable fact can be costly. However, a conservative graph may appear clean while omitting important knowledge. Reports should present both accepted precision and rejected-candidate analysis. Validation thresholds should be selected for the intended use rather than optimized only for F1.

### 8.6 Domain knowledge remains necessary

General models need domain classes, relation meanings, aliases, and evaluation criteria. Biomedical linking uses specialist resources (Zheng et al., 2015). Maintenance graphs use ISO-based labels and correction dictionaries (Stewart et al., 2024). Threat-intelligence systems rely on MITRE ATT&CK and domain entity types (Hua et al., 2023; Yang et al., 2026). Cultural-heritage construction depends on an ontology and Wikidata linking (Schimmenti et al., 2025). LLMs reduce some annotation and coding effort, but they do not eliminate domain modeling.

## 9. Design guidance for practical systems

The corpus supports a simple architecture for new Text-to-KG projects.

### 9.1 Define the graph contract first

Specify which entities, relations, identifiers, provenance fields, and constraints the graph needs. Decide whether new relation types are allowed. Define how uncertainty and conflicting facts will be represented. This prevents a fluent generator from silently defining the graph's semantics.

### 9.2 Separate candidate generation from graph admission

Use rules, encoders, OpenIE, LLMs, or combinations to generate candidate facts. Preserve the source text, document location, model version, prompt, and confidence for each candidate. Apply entity resolution, schema checks, and factual validation before admission. This follows the evidence from confidence estimation, fusion, G-T2KG, news validation, and EDC (Li and Grishman, 2013; Dong et al., 2014; Kabal et al., 2024; Fernández Cañellas, 2023; Zhang and Soh, 2024a).

### 9.3 Use the least complex method that meets the requirement

Stable, narrow tasks may be served by rules or a small supervised encoder. Broad or changing schemas may justify prompting. Repeated high-volume tasks may justify fine-tuning a smaller model. A generative RDF model is appropriate only if serialization validity is tested. The corpus provides no evidence that the largest available LLM is always the best option (Ringwald et al., 2026; Gajo and Barrón-Cedeño, 2025).

### 9.4 Evaluate at every consequential boundary

Measure entities, relations, linking, canonicalization, and final graph quality separately. Add a realistic end-to-end sample. Report minority classes, long documents, unseen relations, and failures rather than only averages. Use human review where correctness cannot be established automatically.

### 9.5 Make privacy and operating cost part of the method

External LLM validation may expose source text (Kabal et al., 2024). Multi-call pipelines add cost (Zhang and Soh, 2024a). Local or open models may reduce privacy risk but require hardware and careful adaptation. A complete methodology should state where data travel, which services are called, the expected number of calls per document, and how failures are handled.

### 9.6 Preserve provenance and support correction

Every accepted graph fact should retain its source and extraction context. Provenance makes manual correction, model auditing, and later updating possible. It also allows downstream users to distinguish an extracted statement from a verified fact. Event-centered named graphs and source-linked triples in the corpus provide useful examples (Martinez-Rodriguez et al., 2018; Fernández Cañellas, 2023).

## 10. Open problems and research agenda

### 10.1 Common evaluation across the full pipeline

The field lacks a benchmark that jointly evaluates source selection, entities, coreference, linking, relations, schema alignment, graph validity, fusion, and updates. Existing datasets isolate valuable tasks but make end-to-end claims difficult. A useful benchmark should include realistic documents, a versioned ontology, evolving facts, provenance, NIL entities, contradictory statements, and explicit matching rules.

### 10.2 Long-document and cross-document reasoning

Sentence-level extraction misses evidence spread across paragraphs or documents. Larger context windows can add noise and cost. Research should evaluate retrieval, discourse structure, coreference, and evidence aggregation together. The goal is not simply more tokens, but selection of the right evidence (Dong et al., 2021; Lairgi et al., 2024; Hua et al., 2023).

### 10.3 Canonical identity under change

Entity and relation resolution remain major barriers to coherent graphs. Systems must handle aliases, abbreviations, emerging entities, changed names, and genuinely distinct but similar concepts. LLM judgments should be paired with stable identifiers, lexical evidence, graph context, and reversible merge decisions (Fernández Cañellas, 2023; Zhang and Soh, 2024a; Mo et al., 2025).

### 10.4 Temporal and event-aware updating

Most studies construct a static graph. Real knowledge changes. A future system should distinguish a new fact from a correction, preserve validity intervals, identify events, and retain conflicting evidence. The limited updating evidence in dynamic traditional Chinese medicine and news-stream graphs shows feasibility but not a general solution (Xu et al., 2022; Fernández Cañellas, 2023).

### 10.5 Calibrated uncertainty and abstention

LLMs can produce plausible but unsupported triples. Binary acceptance hides degrees of uncertainty. Systems need calibrated confidence linked to error consequences, explicit abstention, and routing of uncertain cases to people. Earlier confidence and fusion methods provide foundations that should be integrated with modern generators (Li and Grishman, 2013; Dong et al., 2014).

### 10.6 Privacy-preserving construction

Sending confidential text to an external model is unacceptable in some domains (Kabal et al., 2024). Research should compare local models, protected deployment, selective redaction, and hybrid processing under both accuracy and privacy measures. Privacy cannot be discussed only as a deployment detail when it changes which model and validation method can be used.

### 10.7 Reproducible LLM evaluation

LLM studies require exact model versions, prompts, examples, generation parameters, parsing logic, and timestamps. They should test repeated runs and report variance. Possible benchmark contamination should be assessed with new or fictional data, as attempted by EDC (Zhang and Soh, 2024a). Cost and latency should be reported alongside accuracy.

### 10.8 Human–system collaboration

Human review is often present but weakly specified. Future systems should identify which decisions need experts, measure review time, record disagreement, and learn from corrections without overwriting provenance. The goal is not to remove people from all decisions, but to use their attention where uncertainty and consequence are highest.

## 11. Answers to the review questions

**Which methodologies exist?** The acquired corpus supports twelve families, ranging from rules and dependency extraction to classical learning, neural and transformer models, joint and generative extraction, prompted and adapted LLMs, ontology guidance, hybrid systems, and validation or fusion.

**Which stages do they solve?** Extraction families primarily solve entity and relation recognition. Linguistic and joint methods also address argument structure and overlap. Ontology-guided methods address canonical semantics. Linking and fusion methods address identity and deduplication. Hybrid systems provide the broadest stage coverage, but no family covers preprocessing through temporal maintenance equally well.

**How did the field evolve?** The field moved from explicit linguistic constraints to learned features and contextual representations, then to joint and generative output, and finally to instruction-based LLM systems. The evolution is cumulative: current systems reuse rules, schemas, retrieval, and validation rather than discarding them.

**How are systems evaluated?** Most use precision, recall, and F1 for entities, relations, or triples. Others use ranking metrics, RDF validity, ontology compliance, confidence, human judgment, downstream performance, runtime, cost, or emissions. Incompatible tasks and matching rules prevent a single ranking of all methods.

**What works well?** Rules offer control; linguistic methods offer interpretable structure; supervised models perform well on matched tasks; joint methods handle dependent predictions; generative methods produce flexible structures; LLMs adapt rapidly; ontologies improve interoperability; and validation or fusion raises trustworthiness.

**What are the weaknesses?** Rules are narrow; parsers and pipelines propagate errors; supervised systems require reliable labels; joint methods commonly assume fixed schemas; generators can emit invalid or unsupported output; LLMs are prompt-sensitive and costly; ontologies are incomplete; and validation can reduce recall.

**How can systems improve?** The strongest evidence supports hybrid designs with explicit graph contracts, candidate generation, provenance, identity resolution, schema checks, calibrated confidence, and human review for uncertain cases. Evaluation should include component and graph-level outcomes.

**What remains unresolved?** Major open problems are long-document evidence, cross-document identity, temporal updating, event extraction, calibrated uncertainty, privacy, reproducibility, consistent graph-level evaluation, and efficient human oversight.

## 12. Conclusion

Constructing a knowledge graph from text is not one extraction problem. It is a sequence of decisions about meaning, identity, structure, evidence, and change. The 83-paper corpus shows substantial progress in each part. Rule-based and linguistic systems established transparent representations and scalable OpenIE. Classical learning introduced weak supervision, ranking, confidence, and fusion. Neural and transformer models improved contextual extraction. Joint and generative methods reduced rigid module boundaries. LLMs made it possible to specify new tasks and schemas with instructions or limited adaptation.

The evidence does not justify the claim that LLMs have solved Text-to-KG construction. Their flexibility creates new requirements for canonicalization, factual validation, privacy control, and reproducibility. Nor does the evidence support dismissing earlier methods. Rules, ontologies, linking algorithms, confidence models, and deterministic validators are precisely the components that make modern generators usable.

The most defensible direction is therefore controlled generation: broad candidate extraction combined with explicit identity, schema, provenance, validation, and update policies. Progress should be measured not only by triple F1 on a static benchmark, but by whether a system produces a coherent, auditable, maintainable graph from realistic documents. That shift—from generating plausible triples to managing justified knowledge—is the next step for the field.

## References: acquired and analyzed corpus

The bibliography contains only papers from the acquired and analyzed corpus. `n.d.` indicates that the accepted Paper Card did not report a publication year.

- Abolhasani, M. S., and Pan, R. (2024). *Leveraging LLM for Automated Ontology Extraction and Knowledge Graph Generation*.
- Al-Moslmi, T., Gallofré Ocaña, M., Opdahl, A. L., and Veres, C. (2020). *Named Entity Extraction for Knowledge Graphs: A Literature Overview*.
- Angeli, G., Zhong, V., Chen, D., Chaganty, A., Bolton, J., Premkumar, M. J., Pasupat, P., Gupta, S., and Manning, C. D. (2015). *Bootstrapped Self Training for Knowledge Base Population*.
- Augenstein, I., Maynard, D., and Ciravegna, F. (2016). *Distantly Supervised Web Relation Extraction for Knowledge Base Population*.
- Azarbonyad, H., Zhu, Z. L., Cheirmpos, G., Afzal, Z., Yadav, V., and Tsatsaronis, G. (2025). *Question-Answer Extraction from Scientific Articles Using Knowledge Graphs and Large Language Models*.
- Baek, H.-Y., Choi, J., Seo, J., Jin, X., Lee, D., and Oh, B. (2025). *Relation-Faceted Graph Pooling with LLM Guidance for Dynamic Span-Aware Information Extraction*.
- Bai, X., He, S., Li, Y., Xie, Y., Zhang, X., Du, W., and Li, J.-R. (2025). *Construction of a Knowledge Graph for Framework Material Enabled by Large Language Models and Its Application*.
- Bektemyssova, G., Sabdenov, A., Satybaldiyeva, R., Bykov, A., and Ali, N. (2026). *Optimizing Syntactic-Semantic Relation Extraction for the Kazakh Language with Transformer Architectures and Synthetic Corpora*.
- Bi, Z., Chen, J., Jiang, Y., Xiong, F., Guo, W., Chen, H., and Zhang, N. (2024). *CodeKGC: Code Language Model for Generative Knowledge Graph Construction*.
- Bottino, G. B., and Alcázar, J. J. P. (2025). *Constructing Knowledge Graphs from Text Using Large Language Models: Scoping Review*.
- Bundschus, M. (2010). *From Text to Knowledge: Bridging the Gap with Probabilistic Graphical Models*.
- Bytyçi, A., Ramosaj, L., and Bytyçi, E. (2023). *Review of Automatic and Semi-Automatic Creation of Knowledge Graphs from Structured and Unstructured Data*.
- Cai, H., Liao, W., Liu, Z., Zhang, Y., Huang, X., Ding, S., Ren, H., Wu, Z., Dai, H., Li, S., Wu, L., Liu, N., Li, Q., Liu, T., and Li, X. (2023). *Coarse-to-Fine Knowledge Graph Domain Adaptation Based on Distantly-Supervised Iterative Training*.
- Chaganty, A. T., Paranjape, A. P., Liang, P., and Manning, C. D. (n.d.). *Importance Sampling for Unbiased On-Demand Evaluation of Knowledge Base Population*.
- Chepurova, A., Kuratov, Y., Bulatov, A., and Burtsev, M. (2024). *Prompt Me One More Time: A Two-Step Knowledge Extraction Pipeline with Ontology-Based Verification*.
- Dang, M.-H., Pham, T. H. T., Molli, P., Skaf-Molli, H., and Gaignard, A. (2025). *LLM4Schema.org: Generating Schema.org Markups with Large Language Models*.
- Dong, K. (2024). *Incorporating Contexts to Open Information Extraction*.
- Dong, K., Zhao, Y., Sun, A., Kim, J.-J., and Li, X. (2021). *DocOIE: A Document-Level Context-Aware Dataset for OpenIE*.
- Dong, X. L., Gabrilovich, E., Heitz, G., Horn, W., Lao, N., Murphy, K., Strohmann, T., Sun, S., and Zhang, W. (2014). *Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion*.
- Dredze, M., McNamee, P., Rao, D., Gerber, A., and Finin, T. (2010). *Entity Disambiguation for Knowledge Base Population*.
- Elkhammash, E., and Ben Abdessalem, W. (2019). *A Holy Quran Ontology Construction with Semi-Automatic Population*.
- Exner, P., and Nugues, P. (2012). *Entity Extraction: From Unstructured Text to DBpedia RDF Triples*.
- Fader, A., Soderland, S., and Etzioni, O. (2011). *Identifying Relations for Open Information Extraction*.
- Fernández Cañellas, D. (2023). *Knowledge Graph Population from News Streams*.
- Fossati, M., Dorigatti, E., and Giuliano, C. (2015). *N-ary Relation Extraction for Simultaneous T-Box and A-Box Knowledge Base Augmentation*.
- Freitas, A., Carvalho, D. S., da Silva, J. C. P., O'Riain, S., and Curry, E. (2012). *A Semantic Best-Effort Approach for Extracting Structured Discourse Graphs from Wikipedia*.
- Gajo, P., and Barrón-Cedeño, A. (2025). *Natural vs Programming Language in LLM Knowledge Graph Construction*.
- Gashteovski, K., Gemulla, R., Kotnis, B., Hertling, S., and Meilicke, C. (2020). *On Aligning OpenIE Extractions with Knowledge Bases: A Case Study*.
- Gashteovski, K., Wanner, S., Hertling, S., Broscheit, S., and Gemulla, R. (2019). *OPIEC: An Open Information Extraction Corpus*.
- Getman, J., Ellis, J., Strassel, S., Song, Z., and Tracey, J. (2018). *Laying the Groundwork for Knowledge Base Population: Nine Years of Linguistic Resources for TAC KBP*.
- Hernandez, J., Martinez-Rodriguez, J. L., Lopez-Arevalo, I., Rios-Alvarado, A. B., and Aldana-Bobadilla, E. (2021). *FEEL: Framework for the Integration of Entity Extraction and Linking Systems*.
- Hoffmann, R., Zhang, C., Ling, X., Zettlemoyer, L., and Weld, D. S. (2011). *Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations*.
- Hong, Z., and Huang, H. (2026). *A Survey on Generative Knowledge Graph Construction*.
- Hua, Y., Zou, F., Han, J., Sun, X., and Wang, Y. (2023). *LLM-TIKG: Threat Intelligence Knowledge Graph Construction Utilizing Large Language Model*.
- Ji, H., and Grishman, R. (2011). *Knowledge Base Population: Successful Approaches and Challenges*.
- Kabal, O., Harzallah, M., Guillet, F., and Ichise, R. (2024). *Enhancing Domain-Independent Knowledge Graph Construction through OpenIE Cleaning and LLMs Validation*.
- Kaverinskiy, V. (2025). *Large Language Models for Automatic Knowledge Graph Construction from Text Documents: A Materials Science Case Study*.
- Lairgi, Y., Moncla, L., Cazabet, R., Benabdeslem, K., and Cléau, P. (2024). *iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models*.
- Li, X., and Grishman, R. (2013). *Confidence Estimation for Knowledge Base Population*.
- Lin, X., Li, H., Xin, H., Li, Z., and Chen, L. (2020). *KBPearl*.
- Liu, X., Zhao, W., and Ma, H. (2022). *Research on Domain-Specific Knowledge Graph Based on the RoBERTa-wwm-ext Pretraining Model*.
- Luan, Y., He, L., Ostendorf, M., and Hajishirzi, H. (2018). *Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction*.
- Martinez-Rodriguez, J. L., Hogan, A., and Lopez-Arevalo, I. (2016). *Information Extraction Meets the Semantic Web: A Survey*.
- Martinez-Rodriguez, J. L., Lopez-Arevalo, I., and Rios-Alvarado, A. B. (2018). *OpenIE-based Approach for Knowledge Graph Construction from Text*.
- Mendes, P. N., Jakob, M., García-Silva, A., and Bizer, C. (2011). *DBpedia Spotlight: Shedding Light on the Web of Documents*.
- Mesquita, F., Cannaviccio, M., Schmidek, J., Mirza, P., and Barbosa, D. (2019). *KnowledgeNet: A Benchmark Dataset for Knowledge Base Population*.
- Mihindukulasooriya, N., Tiwari, S., Enguix, C. F., and Lata, K. (2023). *Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text*.
- Milne, D., and Witten, I. H. (2008). *Learning to Link with Wikipedia*.
- Mo, B., Yu, K., Kazdan, J., Cabezas, J., Mpala, P., Yu, L., Cundy, C., Kanatsoulis, C., and Koyejo, S. (2025). *KGGen: Extracting Knowledge Graphs from Plain Text with Language Models*.
- Nayak, T., and Ng, H. T. (2020). *Effective Modeling of Encoder–Decoder Architecture for Joint Entity and Relation Extraction*.
- Regino, A. G., Rossanez, A., Torres, R. S., and dos Reis, J. C. (2024). *A Systematic Literature Review on RDF Triple Generation from Natural Language Texts*.
- Riedel, S., Yao, L., McCallum, A., and Marlin, B. M. (2013). *Relation Extraction with Matrix Factorization and Universal Schemas*.
- Ringwald, C., Gandon, F., Faron, C., Michel, F., and Abi Akl, H. (2026). *Extensive Benchmark of Frugal Encoder–Decoder Language Models for Datatype Properties Extraction and RDF Knowledge Graph Generation*.
- Rios-Alvarado, A. B., Martinez-Rodriguez, J. L., Garcia-Perez, A. G., Guerrero-Melendez, T. Y., Lopez-Arevalo, I., and Gonzalez-Compean, J. L. (2022). *Exploiting Lexical Patterns for Knowledge Graph Construction from Unstructured Text in Spanish*.
- Rossanez, A., dos Reis, J. C., Torres, R. S., and de Ribaupierre, H. (2020). *KGen: A Knowledge Graph Generator from Biomedical Scientific Literature*.
- Salman, M., Haller, A., Rodríguez Méndez, S. J., and Naseem, U. (2024). *Doc-KG: Unstructured Documents to Knowledge Graph Construction, Identification and Validation with Wikidata*.
- Schimmenti, A., Pasqual, V., Vitali, F., and van Erp, M. (2025). *Knowledge Graphs Generation from Cultural Heritage Texts: Combining LLMs and Ontological Engineering for Scholarly Debates*.
- Shang, Y.-M., Huang, H., and Mao, X.-L. (2022). *OneRel: Joint Entity and Relation Extraction with One Module in One Step*.
- Shen, W., Wang, J., and Han, J. (2015). *Entity Linking with a Knowledge Base: Issues, Techniques, and Solutions*.
- Shen, W., Wang, J., Luo, P., and Wang, M. (2012). *LINDEN: Linking Named Entities with Knowledge Base via Semantic Knowledge*.
- Specia, L., and Motta, E. (2006). *A Hybrid Approach for Extracting Semantic Relations from Texts*.
- Stewart, M., and Liu, W. (2020). *Seq2KG: An End-to-End Neural Model for Domain Agnostic Knowledge Graph Construction from Text*.
- Stewart, M., Enkhsaikhan, M., and Liu, W. (2019). *ICDM 2019 Knowledge Graph Contest: Team UWA*.
- Stewart, M., Hodkiewicz, M., Liu, W., and French, T. (2024). *MWO2KG and Echidna: Constructing and Exploring Knowledge Graphs from Maintenance Data*.
- Suchanek, F. M. (2009). *Automated Construction and Growth of a Large Ontology*.
- Wang, S., Zhang, Y., Che, W., and Liu, T. (2018). *Joint Extraction of Entities and Relations Based on a Novel Graph Scheme*.
- Wei, Z., Su, J., Wang, Y., Tian, Y., and Chang, Y. (2020). *A Novel Cascade Binary Tagging Framework for Relational Triple Extraction*.
- Wu, G., He, Y., and Hu, X. (2018). *Entity Linking: An Issue to Extract Corresponding Entity with Knowledge Base*.
- Xia, Y., Zheng, Z., Meng, Y., and Sun, J. (2019). *Semi-Automatic Knowledge Graph Construction by Relation Pattern Extraction*.
- Xu, T., Chen, G., Du, L., Xu, J., Peng, Z., Xin, F., and Li, M. (2022). *A Method for Traditional Chinese Medicine Knowledge Graph Dynamic Construction*.
- Yang, X., Zhong, R., Chen, Y., Peng, G., Yao, D., Chen, C., Wang, C., Zhang, D., Zhou, Y., and Yang, Z. (2026). *CTI-Thinker: An LLM-Driven System for CTI Knowledge Graph Construction and Attack Reasoning*.
- Yang, Y., Yu, K., Gao, S., Yu, S., Xiong, D., Qin, C., Chen, H., Tang, J., Tang, N., and Zhu, H. (2024). *Alzheimer's Disease Knowledge Graph Enhances Knowledge Discovery and Disease Prediction*.
- Ye, H., Zhang, N., Chen, H., and Chen, H. (2022). *Generative Knowledge Graph Construction: A Review*.
- Yu, D., Huang, L., and Ji, H. (2017). *Open Relation Extraction and Grounding*.
- Zaratiana, U., Tomeh, N., Holat, P., and Charnois, T. (2024). *An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction*.
- Zhang, B., and Soh, H. (2024a). *Extract, Define, Canonicalize: An LLM-Based Framework for Knowledge Graph Construction*.
- Zhang, Y., Sadler, T., Taesiri, M. R., Xu, W., and Reformat, M. Z. (2024b). *Fine-Tuning Language Models for Triple Extraction with Data Augmentation*.
- Zhao, W., Chen, Q., and You, J. (2023). *LlmRe: A Zero-Shot Entity Relation Extraction Method Based on the Large Language Model*.
- Zheng, J. G., Howsmon, D., Zhang, B., Hahn, J., McGuinness, D., Hendler, J., and Ji, H. (2015). *Entity Linking for Biomedical Literature*.
- Zheng, Y., Liu, W., Zeng, B., Feng, Y., Du, X., Zhou, L., and Li, Y. (2026). *Automating Biomedical Knowledge Graph Construction for Context-Aware Scientific Inference*.
- Zhong, L., Wu, J., Li, Q., Peng, H., and Wu, X. (2022). *A Comprehensive Survey on Automatic Knowledge Graph Construction*.
- Zhou, S. (2023). *Semi-Supervised Generative Open Information Extraction*.
- Zhu, Y., Wang, X., Chen, J., Qiao, S., Ou, Y., Yao, Y., Deng, S., Chen, H., and Zhang, N. (2024). *LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities*.
