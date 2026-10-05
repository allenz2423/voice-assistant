# Adam memory proposal: evidence, facts, and dates

> **Research update (October 5, 2026):** The deeper review in [agent-memory-research-2026-10-05.md](agent-memory-research-2026-10-05.md) makes the storage decision conditional. Preserve original episodes as answer evidence, add short-lived task state and separately governed procedural lessons, and measure memory-guided actions as well as recall. SQLite should be compared with JSON only after the behavior and resource baselines are fixed.

## Recommendation

Give Adam a small, local memory system with three layers:

1. **Original statements** preserve exactly what the user explicitly asked Adam to remember.
2. **Derived facts and events** make dates, preferences, and corrections queryable without changing the original statement.
3. **Search indexes** find relevant records quickly and can be rebuilt from the first two layers.

SQLite is a possible local home for these layers, but the data model and recall behavior matter more than the file format. Build and evaluate the model against the current JSON store before considering a storage switch. Keep the system useful with lexical search alone; load the embedding model only when semantic recall needs it.

The proposed four fields—Date Created, Memory Title, Memory Content, Relative Date—are a useful display view. They are too small as the underlying record: a relative date needs a capture time, timezone, normalized date or range, and the original phrase. A title should help a person browse memories, not become the evidence Adam relies on when answering.

## Current starting point

[`MemoryRecord`](../src/memory/manager.py) already stores an ID, original `text`, category, creation and update timestamps, access metadata, optional 384-dimensional vector, and optional event type, date range, time range, original time phrase, timezone, and precision. The manager saves a version 2 JSON file by atomic replacement, loads the records at startup, and builds in-memory BM25, vector, and event-date indexes. Explicit memory commands and the existing memory tool are the save paths.

This is a working structured store. The synthetic 10,001-record check in [`PLAN.md`](PLAN.md) measured roughly 116 ms to load on the laptop and about 0.05 ms for date-bounded lookup. Those figures do not establish a storage bottleneck. They also do not measure concurrent updates, long-running memory use, or real recall quality.

Two limitations are worth addressing regardless of storage choice:

- One record has one text field and at most one parsed event. A statement with several facts cannot represent their separate dates or corrections cleanly.
- For undated memories, a very similar embedding can cause a new save to replace an older record. Similar wording is useful for finding possible duplicates, but should not by itself decide that two claims have the same meaning. A correction or negation could otherwise lose its history.

## The memory model

### 1. Capture: preserve the user's words

Every explicit save creates a **source** with a stable ID, verbatim content, capture timestamp, capture timezone, category, and optional display title. Store whether the save came from a direct voice command or the memory tool. Never rewrite the source merely because a parser or model later changes its interpretation.

Titles can be generated for browsing, but they are optional and editable. Retrieval uses the source content and verified facts, not the title alone. Ordinary conversation continues to stay out of durable memory unless the user explicitly asks to save it or separately enables an automatic memory policy.

### 2. Interpret: derive zero or more facts

A source may yield several **facts**. Each fact points back to the source and carries:

| Field | Purpose |
| --- | --- |
| `kind` | `event`, `preference`, `profile`, or `note`; use a narrow kind only when supported by the statement. |
| `statement` | A short, faithful claim suitable for search and display. |
| `status` | `active`, `superseded`, or `retracted`. |
| `supersedes_fact_id` | Links an explicit correction to the earlier claim. |
| `event_date_start`, `event_date_end` | The actual date or range the statement refers to, when known. |
| `event_start_at`, `event_end_at` | Timezone-aware instants when the user supplied an unambiguous time range. |
| `time_expression`, `time_zone`, `precision` | The original phrase, its capture timezone, and whether the interpretation is a day, week, month, year, or exact time. |

Facts without a date remain undated. If a date is ambiguous, keep the original phrase and leave the uncertain fields empty. Do not invent a day for “last year,” or silently shift a nonexistent daylight-saving clock time.

For example, if a user in New York says on October 5, 2026, “Remember that I worked yesterday from 3:45 to 8 pm,” Adam keeps those words in the source. An event fact records October 4, 2026, 3:45–8:00 pm in `America/New_York`, with “yesterday” retained as the original date phrase. The saved timestamp remains October 5. Asking “When did I work last week?” uses the event date, not the save date.

### 3. Correct and forget explicitly

“Actually, I worked until 9 pm” should create a new sourced fact and mark the old fact superseded when Adam can identify the intended shift. If it cannot identify which shift, it should ask rather than update an arbitrary record. Similarity can propose a candidate correction; it cannot authorize one.

“Forget that” or deletion by ID removes the source, its facts, indexes, and embeddings. Superseded facts remain available for an audit of corrections until the user asks to forget them. Answers use active facts by default.

## Retrieval path

1. **Understand the question.** Detect date ranges, event kind, and explicit entities where possible. Keep the parser conservative; an ordinary question should not receive unrelated memories because it shares a generic word.
2. **Filter before ranking.** Use indexed dates and kinds for questions such as “What times did I work this week?” Exclude retracted and superseded facts from ordinary answers.
3. **Find candidates.** Search original content and fact statements with lexical full-text search. Use exact names, numbers, and phrase coverage to prevent a loosely related high-scoring result from entering the prompt.
4. **Use semantics when useful.** If lexical and date search do not provide enough candidates, use the existing optional embedding route. Keep vectors in compact binary form and load a bounded matrix or candidate set on demand. A database alone does not reduce RAM if Adam still loads every index and model at startup.
5. **Return evidence, not a pile of notes.** Pass a bounded set of active facts with their source IDs, verbatim supporting text, and resolved dates to the answer path. If sources conflict or the answer is absent, say so. Measure whether the final answer is correct, not just whether search returned a row.

This keeps the current fast date filtering and optional semantic recall, while adding a way to represent multiple facts and corrections. A display title helps browsing; it should have little weight in answer selection.

## Storage choice

If measured update, concurrency, or resident-memory needs justify a storage change, prototype **SQLite** locally after the model is defined. A compact layout would be:

```text
memory_sources(id, content, title, category, captured_at, captured_timezone, origin)
memory_facts(id, source_id, kind, statement, status, supersedes_fact_id,
             event_date_start, event_date_end, event_start_at, event_end_at,
             time_expression, time_zone, precision)
memory_embeddings(fact_id, model_id, vector_blob)   # optional, derived
memory_fts                                             # derived text index
```

Index active facts by kind and event date; use foreign keys so source deletion removes its derived data. Keep schema migrations versioned. SQLite [FTS5](https://www.sqlite.org/fts5.html) can provide on-disk lexical candidate search; its ranking must be checked against the current BM25 results rather than assumed equivalent. SQLite [WAL mode](https://www.sqlite.org/wal.html) allows local readers alongside one writer, which is useful if the voice service and a memory browser access the same database. It requires a local filesystem and still serializes writers. An SQLite vector extension is not needed for the first version.

The main expected benefits are incremental transactions, consistent correction/deletion, indexed queries, and the option to stop retaining the whole text index in Python memory. A faster answer or lower RAM use is a hypothesis until measured. The current JSON path remains a reasonable baseline, especially for a small personal store.

## Build and acceptance sequence

1. **Fix the recall contract.** Create a small, synthetic evaluation set covering ordinary facts, preferences, multiple facts in one save, corrections, negation, a missing answer, date/week/year questions, overnight shifts, and daylight-saving edge cases. Score final answers and cited source IDs, including false memories injected into unrelated questions. Keep private memories out of test artifacts.
2. **Add the source/fact model behind the existing memory API.** Preserve the JSON store initially. Keep old records as one source plus their existing temporal fact when present; preserve their IDs, original text, timestamps, and stored date interpretation. Do not resolve an old “yesterday” against migration day.
3. **Test SQLite behind the same API if storage needs justify it.** Import a copy of the JSON file in one transaction, validate counts, IDs, text, date fields, and index consistency, then compare both backends on the same queries. Keep the original JSON file as a rollback copy through cutover. Use restrictive local file permissions and a documented backup procedure.
4. **Measure on laptop first.** Compare answer accuracy, p50/p95 recall time, save/update time, startup time, process PSS/RSS, database size, and peak memory with and without embeddings at 100, 1,000, and 10,001 synthetic records. Include a crash or interrupted-write recovery check. Repeat resource checks on Desky only after its experiment pause is lifted; keep its results separate and its memory use low as well.
5. **Switch only after equivalence and a clear gain.** Require no lost memories, no date/correction regressions, and no increase in irrelevant memory injection. Keep the storage switch reversible until live recall through Adam succeeds. Update the plan and progress report, and commit each verified large checkpoint.

## Decision rule

Adopt SQLite if the prototype improves update safety or resource use while preserving recall quality and latency. If the data model improves answers but SQLite offers no measurable benefit at the current store size, keep the model on JSON and defer the storage migration. The durable design is the source-to-fact relationship; SQLite is the implementation choice to validate.
