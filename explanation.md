# A 15-minute presentation of my literature review

**Paper:** *From Rules to Large Language Models: A Review of Methods for Constructing Knowledge Graphs from Unstructured Text*

**Delivery notes:** Read the paragraphs as a first-person presentation. Allow approximately 15 minutes, including brief pauses. Headings and timings are preparation notes and should not be read aloud.

## 0:00–1:15 — Purpose of the research

Good afternoon. Today I will present my paper, “From Rules to Large Language Models: A Review of Methods for Constructing Knowledge Graphs from Unstructured Text.” I prepared it with my coauthors, Andrej Damjanovski, Trajce Prodanov, and Rade Perovanovikj.

The paper examines how computers turn written information into knowledge graphs. A knowledge graph is an organized network of people, places, objects, or concepts and the relationships between them. Its purpose is to make information easier to connect, search, and use.

I reviewed 83 full-text papers, covering research available through August 2026. I examined how their methods work, how the researchers tested them, and what their results tell us about reliability.

My contribution is to bring this research together in a clear comparison. I considered both older methods and recent systems based on large language models, which are AI models that can follow written instructions and generate answers.

The main question was whether these methods produce information that people can check, correct, and continue using over time. This question guided both my analysis and the writing of the paper.

## 1:15–3:00 — How I conducted the review

I began by defining which research was relevant. I included studies that start with ordinary written language and produce information that can help build a knowledge graph. I also included earlier reviews and papers about testing methods, because they helped explain the development of the field.

I used three sources to find research. Elicit helped me search through questions about particular topics. Connected Papers helped me identify studies related to relevant publications. OpenAlex helped me check publication details and follow references to further work.

For each paper, I examined the same questions. What problem does it address? How does its method work? What information does it produce? How was it tested? What are its main results and limitations?

I organized the answers into individual paper summaries and a comparison spreadsheet. This made it easier to identify patterns across the studies and keep track of the evidence behind each finding. The preparation process included AI assistance for structured summaries. I discussed paper selection and classification with my coauthors.

I grouped the methods into twelve families according to how they work. These include methods based on written rules, grammar, learning from examples, different types of AI models, predefined graph categories, combinations of tools, and checks that correct or combine extracted information.

Each paper received one main family label and additional labels where it used several approaches. I also recorded which tasks it covered, from preparing documents and finding relationships to checking the final information. This helped me compare studies with different aims without treating them as identical systems.

## 3:00–4:15 — How I wrote the paper

When writing the paper, I used the individual summaries as supporting evidence. I then organized the discussion around findings that appeared across several studies.

The paper first explains the research problem and review procedure. It then compares the groups of methods, examines how their results were measured, and develops the main conclusions and future research questions.

I kept a clear distinction between results reported by the original researchers and conclusions I drew from comparing their work. The experimental scores come from the original publications. My contribution is to explain what those scores demonstrate and where their meaning is limited.

I compared numerical results only when the tasks and testing conditions were sufficiently clear. A high score on one test cannot automatically be compared with a score from a different test.

I also examined whether the papers provided enough information for another researcher to repeat the work. This helped me assess the strength of the evidence as well as the reported performance. The aim was to explain both the progress in the field and the problems that remain.

## 4:15–6:00 — Finding one: newer methods still need clear rules and checks

My first finding is that newer methods continue to depend on ideas developed in earlier research.

Older systems often used rules written by people. Later systems learned patterns from examples and became better at using surrounding text to interpret meaning. Large language models made it easier to describe a task through instructions and request an organized answer.

However, greater flexibility did not remove the need for control. The systems still need agreed names, clear relationship categories, and checks on the information they produce.

One paper from the prompted language-model family is “Extract, Define, Canonicalize,” by Zhang and Soh. In this family, the model receives written instructions describing its task. This particular method first finds relationships, then explains their meanings and brings equivalent relationship names together. It shows why generating information must be followed by making that information consistent.

Another paper, “Text2KGBench,” by Mihindukulasooriya and colleagues, belongs to the schema-guided family. A schema defines which categories and relationships a graph can contain. The study examines both whether a system follows those rules and whether it extracts the correct facts.

These are different requirements. Following the allowed structure does not guarantee that the information is correct.

I therefore concluded that the newest model is not automatically the best choice for every task. The method should fit the documents, the available examples, and the level of reliability required. Combining different approaches can help, provided that the complete system is tested carefully.

## 6:00–7:45 — Finding two: success in one task does not prove the final result is reliable

My second finding concerns how researchers measure success.

A system may be good at recognizing names or finding relationships but still make mistakes when combining that information into a graph. It may confuse similar names, create duplicate records, or connect a statement to the wrong item.

One paper from the linguistic extraction family is “OpenIE-based Approach for Knowledge Graph Construction from Text,” by Martinez-Rodriguez and colleagues. Linguistic methods use grammar and sentence structure to find information.

This study reported stronger results for some individual tasks than for complete statements. Only 51 percent of the complete extracted statements were judged correct under its evaluation. The lesson is that testing the individual parts does not replace testing their combined output.

Another paper, “OneRel,” by Shang and colleagues, belongs to the joint extraction family. These methods identify the things mentioned in a text and their relationships together. OneRel performed strongly on its extraction tests. However, those tests did not establish how well it would manage conflicting documents, repeated names, or changing information over time.

When writing my evaluation section, I therefore asked what each reported result actually covered.

I also distinguished between recording a source accurately and proving that the source is true. A system can correctly extract a statement from an unreliable document. Keeping the source attached to the statement allows a person to inspect the evidence and decide how much confidence to place in it.

## 7:45–9:30 — Finding three: instructions, examples, and document selection strongly affect results

My third finding is that language-model performance depends on more than the model itself.

One paper from the prompted language-model family is “Prompt Me One More Time,” by Chepurova and colleagues. Removing examples from its instructions reduced its F1 score from 55 percent to 16 percent. F1 is a measure that balances how much correct information a system finds with how many mistakes it makes.

This large change shows that the instructions and examples must be treated as important parts of the research method.

A paper from the fine-tuned language-model family is “Fine-Tuning Language Models for Triple Extraction with Data Augmentation,” by Zhang and colleagues. Fine-tuning means giving an existing model additional training for a particular task. Despite strong results on one test, two models performed very poorly after moving to a different test collection. Success on familiar material did not guarantee success elsewhere.

Another paper from the prompted language-model family, “iText2KG,” by Lairgi and colleagues, examined how much surrounding text to provide. In its computer-science evaluation, 94 percent of extracted statements were correct when nearby context was used, compared with 83 percent when whole-document context was used.

More text sometimes introduced unrelated information.

Finally, “LLM-TIKG,” by Hua and colleagues, belongs to the fine-tuned language-model family. It found that carefully corrected training examples produced better results for identifying entities than a larger, less carefully checked collection.

Together, these studies show why instructions, document selection, and example quality deserve as much attention as model choice.

## 9:30–11:00 — Finding four: checking removes errors but can also remove useful information

My fourth finding concerns the checks applied after information has been extracted.

These checks can improve correctness, but they may also reject valid statements. It is therefore necessary to measure both the mistakes removed and the useful information lost.

One paper from the hybrid family is “Enhancing Domain-Independent Knowledge Graph Construction through OpenIE Cleaning and LLMs Validation,” by Kabal and colleagues. Hybrid methods combine different types of tools. This study combines language-processing methods with language-model checking.

After checking, the share of returned statements that were correct increased from approximately 59 percent to 72 percent. However, the share of expected correct information recovered fell from approximately 48 percent to 44 percent.

In other words, the accepted information became more accurate, but the result was less complete.

Another paper, “FEEL,” by Hernandez and colleagues, belongs to the validation and knowledge-fusion family. These methods check information and combine results from different sources or systems. FEEL also found that stricter settings could improve correctness while missing more valid items.

It highlights a further concern: agreement between systems is less convincing when they tend to make the same mistakes.

I concluded that checking should be evaluated as carefully as extraction. The paper also emphasizes preserving original sources and allowing decisions to be reversed. If a statement is rejected or two records are combined incorrectly, the evidence should remain available for later correction.

## 11:00–12:30 — Finding five: published results are not always easy to assess or repeat

My fifth finding concerns how clearly the studies report their work.

Some papers test a complete process, while others test only selected parts. They also use different documents, scoring rules, and amounts of human review. These differences make broad comparisons difficult.

One example from the neural language-processing family is “MWO2KG and Echidna,” by Stewart and colleagues. This family uses models that learn language patterns from examples. The study evaluates the recognition of important terms and the classification of equipment failures, but does not provide a numerical assessment of the final graph's correctness.

Its reported results are useful, but they answer a narrower question than whether the complete graph is reliable.

I also checked whether papers reported publicly available software and research data. Approximately one third reported both. Exact language-model versions were not always clearly identified either.

These findings concern what the papers reported. I did not recheck whether every availability link still worked, and sharing files alone does not guarantee that another researcher can repeat an experiment.

My conclusion is that studies need clearer descriptions of their data preparation, model versions, instructions, settings, and testing procedures. Without those details, it is difficult to understand why a method performed well or whether the same result can be obtained again.

## 12:30–13:30 — Limitations and future research

My review also has limits. Its coverage depends on the searches, the starting papers, and access to the full texts. Including all the method families does not mean that every family has equally strong evidence or that every relevant publication was found.

My coauthors and I discussed classification decisions and reached agreement together. We did not rate the papers separately and then measure how often those independent ratings agreed.

The literature provides less evidence about keeping graphs current, representing events, and using consistent procedures for human review.

Future studies should therefore examine more realistic conditions. Information may be spread across documents, sources may disagree, and facts may change. Researchers also need to measure how much effort people spend checking and correcting the results.

These evaluations would help establish whether improvements on research tests lead to information that remains useful in practice.

## 13:30–15:00 — Main conclusions

My overall conclusion is that reliable knowledge graph construction requires several tasks to work together. A system must interpret the text, identify the correct things, represent relationships consistently, preserve evidence, and support later changes.

The reviewed methods have improved different parts of this process. However, no single approach consistently solves every part. Language models offer flexibility, while rules, reference sources, and checking procedures remain important.

The strongest recurring direction is what the paper calls controlled generation. This means allowing a system to propose information, then checking it before accepting it into the graph. Those checks should cover names, relationships, supporting sources, and compatibility with information already stored.

This conclusion reflects a pattern across the reviewed studies. It does not establish that one particular combination of tools is best for every application.

A second conclusion is that research should measure the quality of the final result as well as individual tasks. Correctness, completeness, duplicate information, source records, and the ability to make corrections all matter.

A third conclusion is that clear reporting is essential. Other researchers need enough information to understand the procedure, assess the claims, and repeat the work.

My paper brings these findings together through a consistent comparison of older and newer methods. It explains both where progress has been made and where further evidence is needed.

The main message is that extracting information is only the beginning. The final aim is organized knowledge that people can inspect, correct, and maintain with confidence.

Thank you.
