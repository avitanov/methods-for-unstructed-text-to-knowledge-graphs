# Corpus saturation assessment

## Decision

**Start review synthesis.** The review is based on the 83 accepted Paper Cards, not on all 115 retained candidates. The remaining 31 unavailable PDFs and one manual-review card may add examples, but are not needed to establish the main methodological comparison.

## Current corpus

- Retained candidates: 115
- Accepted Paper Cards: 83
- Manual PDF required: 31
- Paper Card manual review: 1
- Accepted coverage: 72.17%

## Methodology coverage

All 12 methodology families are represented:

1. Rule-based NLP
2. Linguistic and dependency-based extraction
3. Classical machine learning
4. Neural NLP pipelines
5. Transformer encoder methods
6. Joint entity-relation extraction
7. Generative text-to-structure methods
8. Zero-shot and few-shot LLM prompting
9. Instruction-tuned or fine-tuned LLM methods
10. Ontology-constrained and schema-guided methods
11. Hybrid NLP and LLM methods
12. Validation, correction, and knowledge-fusion methods

Rule-based, linguistic, classical ML, joint extraction, generative, prompting, and validation/fusion methods are well represented. Neural pipelines, transformer encoders, ontology/schema-guided methods, and hybrid NLP–LLM methods are adequately represented. Fine-tuned/instruction-tuned LLM methods are represented but less numerous, so conclusions about that subfamily should remain qualified.

## Pipeline coverage

The corpus covers preprocessing, entity and relation/triple extraction, joint extraction, entity linking, coreference, ontology/schema alignment, graph construction, and validation/fusion. Temporal updating and event extraction are less represented; treat them as open evidence gaps rather than claiming comprehensive comparison.

## Evaluation coverage

The cards provide extraction metrics, benchmark datasets, baselines, end-to-end graph outputs, validation/confidence evidence, ablations, and qualitative or expert-reviewed examples. Results must only be compared within compatible task, dataset, split, and metric settings. Human evaluation and reproducibility reporting are uneven and should be reported as limitations.

## Writing guardrails

- Do not generalize from the accepted corpus to all Text-to-KG literature.
- Distinguish source-reported results from review interpretation.
- Qualify claims about fine-tuned LLMs, temporal updating, event extraction, human evaluation, and reproducibility.
- Use the synthesis matrix and individual Paper Cards as the evidence base for every comparative claim.
