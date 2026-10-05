# Agent memory research for Adam — October 5, 2026

## The decision in one paragraph

**SQLite is a plausible local storage engine, not an agent memory architecture.** Adam's next memory improvement should be judged by what it remembers, how it updates or forgets a claim, whether it finds the right evidence, and whether it uses that evidence in an answer or action. Keep the current JSON store as the baseline while testing those behaviors. If incremental writes, reliable multi-process access, or a smaller resident index becomes a measured need, use SQLite as the local source of truth; a separate graph database or always-on vector service has no demonstrated need in this project. This is an inference from the research and Adam's current measurements, not a claim that one architecture has solved agent memory.

## Why the field is still open

The strongest recent benchmarks test different parts of memory and expose different failures:

| Study | What it tests | Relevant finding |
| --- | --- | --- |
| [LongMemEval, ICLR 2025](https://arxiv.org/html/2410.10813) | 500 questions on extraction, multi-session reasoning, temporal reasoning, updates, and abstention. | Memory quality depends on what is indexed, how queries are expanded, and how retrieved evidence is read. Compressing sessions to isolated facts can lose useful context; fact-enriched *search keys* helped without replacing the original record. |
| [MemoryAgentBench, ICLR 2026](https://arxiv.org/html/2507.05257) | Incremental retrieval, learning, long-range understanding, and selective forgetting. | Retrieval methods were strong on finding snippets; long-context methods were stronger on some learning/global tasks. Every evaluated method struggled with selective forgetting; the best multi-hop result was at most 28% in the paper's setup. |
| [MemoryArena, ICML 2026](https://proceedings.mlr.press/v306/he26am.html) | Multi-session tasks whose later actions depend on earlier experiences, including web navigation and preference-constrained planning. | Systems that did well on conversational recall benchmarks still performed poorly when memory had to guide later actions. |
| [Mem2ActBench, ACL 2026](https://aclanthology.org/2026.acl-long.370/) | Whether saved preferences and task state actually ground tool selection and parameters. | Seven evaluated memory frameworks remained weak at using memory during tool tasks. This is directly relevant to Adam's desktop and personal assistant behavior. |
| [BEAM and LIGHT, 2026 preprint](https://arxiv.org/html/2510.27246) | Coherent conversations up to 10 million tokens with 2,000 validated questions. | Even large context windows degrade as history grows; combining episodes, current working memory, and a fact scratchpad helped in this benchmark. |
| [LoCoMo-Plus, 2026 preprint](https://arxiv.org/html/2602.10715) | Applying goals and preferences when the later request shares few words with the original cue. | Explicit factual recall misses behaviorally important constraints. The benchmark is constructed and validated for this purpose; it does not settle how an assistant should obtain consent for durable personal memory. |

These are not interchangeable leaderboards. Some use synthetic conversations, different base models, different ingestion rules, or LLM judges. A high question-answering score does not demonstrate correct tool actions, safe memory writes, real voice latency, or low laptop RAM. Papers from memory vendors also evaluate their own systems, so their scores are useful design evidence but not a neutral product ranking.

## Leading architecture families

| Family and representative work | Useful mechanism | Main gap or cost for Adam |
| --- | --- | --- |
| **Full history / long context** | Preserves complete evidence and avoids extraction loss. It is an important quality baseline. [LongMemEval](https://arxiv.org/html/2410.10813) and [BEAM](https://arxiv.org/html/2510.27246) both test it. | Prompt cost and latency grow with history; relevant details can still be missed. Adam's ordinary interactions have a five-second first-response target. |
| **Flat lexical or vector retrieval** | Simple, local, and already close to Adam's BM25 plus optional 384-dimensional embeddings. Strong for a named fact or exact date. [MemoryAgentBench](https://arxiv.org/html/2507.05257) found retrieval methods effective on accurate retrieval tasks. | Similarity is not truth: it can miss implicit constraints, confuse a correction with an old claim, or retrieve a loosely related note. |
| **Tiered context management** — [MemGPT](https://arxiv.org/abs/2310.08560), [MemoryOS](https://aclanthology.org/2025.emnlp-main.1318/), [LIGHT](https://arxiv.org/html/2510.27246) | Separates current working context from archived episodes and compact summaries. Useful for ongoing multi-step tasks and bounded prompts. | Summaries and self-edits can discard details. More memory operations add model calls and latency; their policies still need evaluation. |
| **Extracted facts with update rules** — [Mem0](https://arxiv.org/html/2504.19413), [Hindsight](https://arxiv.org/html/2512.12818) | Turns dialogue into searchable claims and tracks changes. Hindsight explicitly separates observed facts, experiences, summaries, and opinions. | Extraction can be wrong; a derived fact must remain traceable to the original text. Hindsight's paper identifies tool use, controlled forgetting, and privacy-aware management as future work. |
| **Linked notes and temporal graphs** — [A-MEM](https://arxiv.org/abs/2502.12110), [Zep/Graphiti](https://arxiv.org/html/2501.13956) | Can link people, events, and changing relations. Zep distinguishes when a fact was true from when it entered the store and links graph facts to source episodes. | Entity/fact extraction, deduplication, and graph maintenance add model calls and complexity. In [Mem0's own evaluation](https://arxiv.org/html/2504.19413), its graph extension improved temporal results but did not help every single-hop or multi-hop category. Zep also reported weaker results than full context on assistant-side facts in some LongMemEval slices. |
| **Reflective and procedural memory** — [Generative Agents](https://arxiv.org/abs/2304.03442), [Reflexion](https://arxiv.org/abs/2303.11366), [Voyager](https://arxiv.org/abs/2305.16291) | Stores lessons from experience and reusable successful behaviors, rather than only facts about the user. Potentially useful for repeated desktop workflows. | A lesson derived from an unverified or failed action can teach the wrong behavior. Adam must verify outcome before promoting a trace into a reusable procedure. |

No family wins on every task. MemoryAgentBench directly shows different strengths for retrieval and long-context agents; the mobile-agent architecture comparison also reports that [no tested structure consistently dominated across its short- and long-term settings](https://doi.org/10.1145/3812836.3814752). That comparison is small and mobile-specific, so it is supporting evidence rather than a rule for Adam.

### Findings that change our earlier proposal

1. **Keep source episodes, not just atomic facts.** LongMemEval found that compressing history into isolated facts could hurt overall performance. Facts can improve indexing, but the answer path needs access to the original wording and nearby context. Adam's proposed source/fact split supports this if retrieval actually returns the source when needed.
2. **Separate memory types by use.** A preference that should shape a future answer, a past work shift, the state of an unfinished task, and a verified lesson from a desktop failure have different lifetimes and retrieval triggers. One flat `memories` table or vector index blurs them.
3. **Treat a correction as an evidence update.** An old fact may have been true at an earlier time, or it may have been wrong. Preserve both the recorded time and the time when the statement was valid. A new utterance must not be merged solely because its embedding resembles an older one.
4. **Test use, not just recall.** A system may retrieve “avoid evening notifications” and still schedule a loud evening alert. MemoryArena and Mem2ActBench show why action selection and tool parameter checks belong in the evaluation.
5. **Control the write boundary.** [Memory poisoning research](https://arxiv.org/html/2606.04329) finds persistent attacks become easier as agents write and retrieve memory more aggressively. Browser pages, OCR, terminal output, and documents are untrusted evidence; they should not become authoritative user preferences or operating instructions without the user's deliberate save or a separate approved policy.

## Recommended architecture for Adam

This is a small, evidence-first design that can run on the 16 GB laptop. It borrows mechanisms from several papers without adopting any one framework wholesale.

```text
Current request + verified tool state -> task working memory (short lifetime)
Explicit user save                  -> durable original episode/source
Durable source                      -> optional typed facts/events/preferences
Verified completed task             -> candidate procedural lesson (separate policy)
Sources and derived views           -> lexical/date indexes; optional embeddings
Question or task                    -> routed retrieval -> bounded evidence bundle
Evidence bundle                    -> answer or action -> outcome verification
```

### Memory tiers and authority

| Tier | Contents | Write rule | Lifetime / use |
| --- | --- | --- | --- |
| **Task state** | Current goal, completed steps, observed UI state, pending actions. | From the active task and verified tool outcomes. | Session or task lifetime; never silently treated as a durable personal fact. |
| **Personal episodes** | The user's exact saved statement and its capture context. | Existing explicit “remember” paths. | Durable until corrected or forgotten by the user. Highest authority about what the user said, though the statement may itself be uncertain. |
| **Derived facts** | Dates, entities, preferences, event type, and concise search keys with links to their source. | Deterministic parsing first; model extraction only when useful and checked against source. | Rebuildable; cannot erase the source. Record `observed_at`, `valid_from/to` or event date, timezone, and uncertainty separately. |
| **Procedural lessons** | A reusable lesson or workflow from a completed task. | Promote only from a verified outcome under an explicit future policy. | Separate from user facts; carry scope, host, application, and last validation time. |
| **Summaries/links** | Cached topic summaries and relations. | Derived from cited sources. | Disposable and invalidated when source facts change; never the only copy of a fact. |

The current explicit-save rule should remain the default for durable personal memories. Implicit goals and preferences can still help the active session through task state. Durable automatic extraction is a separate product choice that needs user control, privacy rules, and poisoning tests.

### Retrieval and consumption

Route by need rather than searching every memory the same way:

1. **Exact/time query:** use date and type filters first, then rank the remaining sources. “What times did I work this week?” should use event time, not save time.
2. **Named fact:** use lexical and exact identifier search; preserve numbers and negation. Search the source text as well as derived search keys.
3. **Broad or paraphrased query:** use optional embeddings to widen candidates, then check source support. Keep the embedding model lazy and its resident memory measured.
4. **Preference or action:** look for active, relevant constraints even without word overlap; pass only supported constraints into the tool plan and verify the final action. This is a harder research target, not something a vector query alone guarantees.
5. **Conflicting or missing evidence:** surface the conflict or abstain. Bound the evidence sent to the model and include source IDs, capture times, event times, and status.

This resembles LongMemEval's separation of indexing, retrieval, and reading. A correctly retrieved item can still be misused by the answer model, so score evidence recall and final answer/action separately.

## Where SQLite fits

SQLite can store sources, derived facts, status, temporal indexes, and an optional edge table. [FTS5](https://www.sqlite.org/fts5.html) can generate lexical candidates, and [WAL mode](https://www.sqlite.org/wal.html) permits local readers alongside a writer. A graph does not require a dedicated graph server at Adam's current scale; a small relationship table can be tried if multi-hop queries show a need. Embeddings can remain optional binary vectors in the same local file or a companion index.

| Backing store | What it buys Adam | Cost or limit now |
| --- | --- | --- |
| **Current JSON plus in-memory indexes** | Already implemented, atomic file replacement, quick synthetic date reads, no new dependencies. | Every save rewrites the file, indexes are rebuilt in process, and independent writers would need coordination. |
| **SQLite with FTS5** | Local transactions, date/status indexes, incremental updates, built-in lexical search, and a path to loading fewer records at startup. | Migration and index-maintenance code; dense search still needs an embedding policy. It does not decide which memory is true or relevant. |
| **Dedicated vector service** | Can help at a much larger semantic-search scale. | Extra resident process or network service; exact time and correction semantics still need another store. No current Adam measurement justifies it. |
| **Dedicated graph database** | Rich relation traversal when verified multi-hop failures demand it. | Entity-resolution and edge-maintenance complexity, usually additional service resources. The published graph improvements are uneven across question types. |

That does **not** show SQLite will improve recall or memory use. Adam's present version 2 JSON store has atomic replacement, an in-memory date index, BM25, and optional dense search in [`MemoryManager`](../src/memory/manager.py). The synthetic 10,001-record check documented in [`PLAN.md`](PLAN.md) took about 116 ms to load on the laptop, with date-bounded lookup near 0.05 ms. SQLite earns its complexity only if an end-to-end comparison shows a benefit such as safer incremental updates, effective multi-process access, less process RAM after moving indexes to disk, or simpler correction/deletion integrity.

The next experiment should hold the *memory model and retrieval policy constant* while swapping JSON for SQLite. Otherwise a better answer cannot be attributed to the database. Use local SQLite; no remote vector store or graph service is justified by the current evidence. Keep Desky experiments paused until the user explicitly lifts that pause, and minimize its memory use when testing resumes.

## Evaluation Adam should run before choosing storage

Build a small private synthetic suite and run it through the real Adam answer/tool path with an isolated memory store. Measure:

| Stage | Questions |
| --- | --- |
| **Write** | Was the intended fact stored? Was an incidental, quoted, or untrusted instruction kept out? Did one utterance with two events retain both? |
| **Update/forget** | Does a correction supersede the right claim? Can Adam distinguish “was true then” from “was mistaken”? Does deletion remove derived summaries and vectors too? |
| **Retrieve** | Is the supporting source in top-k? Does a date filter exclude the wrong week? Are names, numbers, negation, and timezone preserved? |
| **Use** | Is the final answer correct and grounded? Does a remembered preference change the right tool parameter without changing unrelated tasks? Does Adam abstain when evidence is absent? |
| **Resources** | p50/p95 write and retrieval latency, first response and total task latency, tokens/cost, startup and peak PSS/RSS, CPU, database size, and concurrent speech/vision memory on the laptop. |

Compare at least four variants on the same examples and model route: current JSON/BM25-plus-optional-dense baseline; source episodes with derived search keys; source episodes plus typed temporal facts and corrections; and that same model backed by SQLite. Test a graph or extra summary tier only when a specific category still fails. Use a small subset of LongMemEval and an Adam-specific action set inspired by MemoryArena/Mem2ActBench. Do not claim a benchmark ranking from a few synthetic examples or LLM-judge scores alone; inspect raw errors and include human-scored cases.

## Practical next steps

1. Freeze a baseline on the current code and synthetic memories, including false retrieval and incorrect action cases.
2. Prototype source-preserving derived search keys and correction semantics behind the existing `MemoryManager` interface while retaining JSON.
3. Add separate task-state and verified procedural-memory experiments only after defining their write rules and scope.
4. Compare SQLite against JSON with identical data and retrieval. Switch storage only if it provides a measured operational gain with no recall regression.
5. Validate through Adam's live path on the laptop without disrupting daytime use. Commit each verified large checkpoint; test Desky only after its pause is lifted.

## Research limits

Most cited systems are evaluated on generated or curated text histories, often with a model judging answers. Their published scores use different models, prompts, budgets, and ingestion rules, so percentages should not be compared across papers as a common leaderboard. Hindsight's strongest results remain conversational evaluations; its authors list tool use and privacy-aware memory as future work. Zep's own LongMemEval table has categories where full context performs better. MemoryArena and Mem2ActBench are stronger evidence that correct future *actions* remain unsolved, but they do not directly measure Adam's voice, desktop environment, or hardware. The architecture above is a research-backed hypothesis for Adam, not a verified product result.
