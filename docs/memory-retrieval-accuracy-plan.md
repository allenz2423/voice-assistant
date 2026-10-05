# Adam memory retrieval accuracy plan — October 5, 2026

## Decision

Improve the chance that Adam **finds the right evidence and uses it correctly**. Keep the current JSON store during this work. A database migration cannot fix a missed memory, an irrelevant match, or an answer that ignores retrieved evidence. Judge each change through Adam's actual answer and tool paths, with retrieval quality and final behavior scored separately.

The [architecture research](agent-memory-research-2026-10-05.md) surveyed memory systems. This document turns its retrieval findings into an experiment sequence for Adam. The [LongMemEval paper](https://arxiv.org/html/2410.10813) provides a useful split: indexing, retrieval, and reading can each fail independently. Its fact-enriched search keys improved retrieval and downstream answers in that benchmark, while overcompressing source context could hurt. Those results motivate experiments here; they do not predict Adam's gains.

## What Adam currently does

[`MemoryManager.search`](../src/memory/manager.py) uses BM25 over saved text, optional embedding similarity, and a token-overlap bonus. It combines dense and normalized BM25 scores with fixed weights (`0.55` and `0.45`) and returns the highest-scoring accepted matches. [`retrieve_context`](../src/memory/manager.py) issues one search for the user's utterance, normally keeps three matches, or keeps at most 20 for a recognized date range. It formats memory text and resolved event times for the model. [`brain.py`](../src/llm/brain.py) places that text into the prompt.

The following are **code-derived failure hypotheses**, not observed production error rates:

| Stage | Risk in the present path | Probe |
| --- | --- | --- |
| Query interpretation | One literal query can miss a paraphrase, alias, or preference that matters to an action but shares few words with the request. | Ask the same question with names, pronouns, paraphrases, and indirect goals. |
| Candidate generation | BM25 indexes raw `record.text`; optional vectors broaden matches, but neither channel represents a verified claim or its current status. | Check whether the gold source appears in the top 3, 10, and 20 before changing the prompt. |
| Ranking | Fixed score weights and a bonus for **any matching number** can rank a wrong date, person, or code highly. There is no explicit negation or entity agreement check. | Add paired distractors that differ by one number, name, or “not.” |
| Acceptance | When a date range is recognized, every candidate in the date range is accepted even if its words are unrelated; event type narrows this only when recognized. | Mix work, travel, reminders, and notes on the same dates. |
| Completeness | Date queries truncate to 20 **before** chronological ordering. A request for all events can omit qualifying records without telling the model. | Save more than 20 qualifying events; inspect omissions and answer completeness. |
| Evidence handoff | The formatted prompt omits source IDs, capture time, score, and correction status. The model cannot reliably distinguish an older claim from a correction using metadata it never receives. | Present conflicting saved statements and inspect both retrieved set and final answer. |
| Answer/action use | Retrieval success does not prove Adam answered from the evidence or applied a preference to a tool parameter. | Score final answer and action outcome separately from top-k recall. |

Some date acceptance is intentional for list queries, where all events in range may matter. The error would be applying that policy to a selective question, or silently truncating a complete list. Do not change the acceptance rule until both cases are in the evaluation set.

## Evaluation contract

Build an isolated, synthetic memory corpus with stable source IDs. Keep a held-out set after tuning. Each case records the query, expected source IDs (possibly an empty set), forbidden distractor IDs, expected answer facts, and, for an action, required tool parameters. Include realistic query wording and multi-turn corrections; do not copy private memories into test fixtures.

| Case family | Required examples |
| --- | --- |
| Exact and paraphrased recall | Same fact asked with its original wording, synonym, alias, and pronoun. |
| Discriminative recall | Same person/topic with different numbers or dates; affirmative versus negative preference; similarly named people. |
| Temporal recall | Relative dates, week boundaries, time zones, overnight events, overlapping ranges, unknown dates, and multiple events on one day. |
| Complete listing | Zero, one, and more than 20 qualifying events; a selective question within a busy date range. |
| Updates and conflicts | Explicit correction, past truth versus current truth, contradictory statements, deletion, and ambiguous correction target. |
| Implicit constraint | A saved preference that should affect a later task with little word overlap. [LoCoMo-Plus](https://arxiv.org/html/2602.10715) probes this kind of latent constraint. |
| No answer | Related distractors exist but no source supports the requested claim. |
| Action use | The correct memory is retrieved; verify that the final tool choice and parameters respect it. [MemoryArena](https://proceedings.mlr.press/v306/he26am.html) and [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/) show why this needs its own score. |

Report **candidate recall@3/10/20**, precision@3, and ranking quality (MRR for a single gold source; nDCG or set coverage for multi-source answers). Separately report unsupported-memory injection rate, no-answer abstention accuracy, answer correctness with source support, list completeness, and action success. Break every measure down by case family: an average can hide a bad failure on corrections or negation. Record p50/p95 retrieval and first-response latency, prompt tokens, CPU, and peak/resident RAM on the laptop. Desktop checks can follow its paused experiment schedule, with low memory use.

Classify each miss before changing architecture: **not saved**, **query misunderstood**, **source absent from candidates**, **source ranked too low**, **wrong evidence accepted**, **evidence missing from prompt**, or **answer/action misused evidence**. This prevents a reranker from being credited for a write failure or a prompt change from being credited for candidate recall.

## Experiment sequence

1. **Freeze a baseline.** Run the current JSON/BM25/optional-vector path on the same fixtures and actual answer/tool route. Capture the candidate IDs and formatted evidence, not just the final text. Label failures manually where scoring is ambiguous.
2. **Improve candidate keys while retaining source text.** Test small, source-linked aliases and fact search keys for names, events, and preferences. Use deterministic extraction first. An extracted key may help find a source; the final answer must still have the original supporting words. Ablate lexical only, current hybrid, and expanded keys. LongMemEval reports gains from fact-enriched keys, but Adam's result must be measured locally.
3. **Route queries by intent.** Distinguish exact lookup, date enumeration, selective date question, broad question, and memory-relevant action. Resolve a date range once against the user's timezone and capture a query trace. For enumeration, retrieve all qualifying bounded records in a controlled, paginated/aggregated path; report if an output limit prevents a complete answer. For selective questions, require topical support in addition to date overlap.
4. **Rerank and verify.** Compare the fixed hybrid weights with a small calibrated reranker using entity/number agreement, negation, event type, temporal overlap, and source status. Keep semantic similarity as a candidate signal, not proof. Never treat one matching digit as exact agreement. Handle an explicit correction before recency is used as a tie breaker. Measure false matches as carefully as recall.
5. **Make the evidence handoff explicit.** Return bounded source IDs, exact excerpts, capture and event times, and correction status to the answer path. If evidence is missing or conflicting, make one targeted second lookup when useful; otherwise abstain or state the conflict. Score whether the final answer cites the right source and whether actions respect the supported constraint. Any extra model call must earn its latency and memory cost.

Change one component at a time and compare it with the frozen baseline on the held-out cases. Prefer deterministic filters for exact facts and dates. Keep any learned or model-based reranking optional and lazy so ordinary voice responses and the 16 GB laptop do not pay its cost when unnecessary. Commit each verified large checkpoint.

## Promotion rule

Promote a change only when it improves the targeted held-out failure family **without worsening** unrelated-question injection, no-answer behavior, existing temporal behavior, or final answer/action quality. Require a source-level explanation for every regression. Report accuracy, latency, and RAM together; do not claim a retrieval improvement from a better prompt alone. The first milestone is a trustworthy error map and baseline; the winning retrieval policy is an empirical result, not a choice of database or memory framework.
