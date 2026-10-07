"""Unit tests for the dynamic memory system."""

import json
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.memory.bm25 import BM25Index, tokenize
from src.memory.embedder import MemoryEmbedder
from src.memory.manager import (
    MemoryManager,
    extract_memory_command,
)
from src.memory.temporal import parse_event_time


def test_bm25_tokenize():
    tokens = tokenize("The quick brown fox jumps over 123 lazy dogs!", filter_stopwords=True)
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "123" in tokens
    assert "dogs" in tokens
    assert "the" not in tokens  # Stopword removed


def test_bm25_scoring():
    index = BM25Index()
    docs = [
        "My wifi password for Guest-Network is SecretPass123",
        "The gate code for the back entrance is 4829#",
        "I love coding in Rust and Python",
    ]
    index.fit(["doc1", "doc2", "doc3"], docs)

    # Search for wifi
    scores = index.score("what is the wifi password?")
    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

    # Search for gate code
    scores = index.score("what was that 4829 gate code?")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_extract_memory_command():
    assert extract_memory_command("remember that my brother's name is Alex") == "my brother's name is Alex"
    assert extract_memory_command("please remember to buy milk on Tuesday") == "buy milk on Tuesday"
    assert extract_memory_command("don't forget that OBS needs display selection") == "OBS needs display selection"
    assert extract_memory_command("save this to memory: the door code is 9988") == "the door code is 9988"
    assert extract_memory_command("make a memory that I prefer dark mode") == "I prefer dark mode"
    assert extract_memory_command("open spotify and play jazz") is None


def test_daemon_loop_can_resolve_memory_command_helper():
    from src.main import extract_memory_command as daemon_extract_memory_command

    assert daemon_extract_memory_command("remember that I prefer dark mode") == "I prefer dark mode"


def test_memory_manager_crud_and_persistence(tmp_path):
    storage = tmp_path / "test_memories.json"
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))

    # 1. Save
    rec1 = mgr.save("My wifi password is SecretPass123", category="credentials")
    rec2 = mgr.save("The front gate code is 4829#", category="home")
    rec3 = mgr.save("My favorite code editor is Zed", category="preferences")

    assert len(mgr.list_memories()) == 3
    assert storage.is_file()

    # 2. Search
    results = mgr.search("what is my wifi password?")
    assert len(results) >= 1
    assert results[0].id == rec1.id
    assert "SecretPass123" in results[0].text

    # Search exact number
    results2 = mgr.search("what is the gate code 4829?")
    assert len(results2) >= 1
    assert results2[0].id == rec2.id

    # 3. Context retrieval
    ctx = mgr.retrieve_context("tell me the wifi password")
    assert ctx is not None
    assert "SecretPass123" in ctx

    # 4. Update
    mgr.update(rec1.id, "My wifi password was changed to NewPass999")
    ctx_updated = mgr.retrieve_context("what is the wifi password?")
    assert "NewPass999" in ctx_updated

    # 5. Delete
    mgr.delete(rec2.id)
    assert len(mgr.list_memories()) == 2
    assert mgr.search("gate code 4829") == []


def test_memory_manager_hybrid_with_mock_embeddings(tmp_path):
    storage = tmp_path / "test_hybrid.json"

    # Mock embedder that creates distinct 384-d vectors
    def mock_encoder(texts):
        vecs = []
        for t in texts:
            val = hash(t) % 1000 / 1000.0
            v = np.ones(384, dtype=np.float32) * val
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.asarray(vecs, dtype=np.float32)

    mgr = MemoryManager(
        storage_path=storage,
        encoder_fn=mock_encoder,
    )

    rec1 = mgr.save("I always drink iced matcha lattes in the afternoon")
    rec2 = mgr.save("The wifi password is SuperSecret!")

    # Search keyword
    res = mgr.search("wifi password")
    assert len(res) >= 1
    assert res[0].id == rec2.id


def test_vector_memory_search_scores_full_and_date_filtered_candidates(tmp_path):
    vector_today = np.zeros(384, dtype=np.float32)
    vector_today[0] = 1.0
    vector_yesterday = np.zeros(384, dtype=np.float32)
    vector_yesterday[1] = 1.0
    vector_pasta = np.zeros(384, dtype=np.float32)
    vector_pasta[2] = 1.0

    def mock_encoder(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                vector_today if "today" in lowered
                else vector_pasta if "pasta" in lowered
                else vector_yesterday if "yesterday" in lowered
                else vector_today
            )
        return np.asarray(vectors, dtype=np.float32)

    stored = {
        "version": 2,
        "memories": [
            {
                "id": "today",
                "text": "I worked today from 3:45 to 8:00pm",
                "event_type": "work",
                "event_date_start": "2026-10-04",
                "event_date_end": "2026-10-04",
                "vector": vector_today.tolist(),
            },
            {
                "id": "yesterday",
                "text": "I worked yesterday from 9:00am to 5:00pm",
                "event_type": "work",
                "event_date_start": "2026-10-03",
                "event_date_end": "2026-10-03",
                "vector": vector_yesterday.tolist(),
            },
        ],
    }
    storage = tmp_path / "vector_memories.json"
    storage.write_text(json.dumps(stored), encoding="utf-8")
    mgr = MemoryManager(
        storage_path=storage,
        encoder_fn=mock_encoder,
    )
    assert mgr._vector_matrix.shape == (2, 384)
    assert all(memory.vector is None for memory in mgr.list_memories())

    full_results = mgr.search("worked", limit=2)
    assert {result.id for result in full_results} == {"today", "yesterday"}
    scores_by_id = {result.id: result.dense_score for result in full_results}
    assert scores_by_id["today"] == pytest.approx(1.0)
    assert scores_by_id["yesterday"] == pytest.approx(0.0)

    dated_results = mgr.search(
        "worked",
        limit=2,
        date_range=("2026-10-04", "2026-10-04"),
        event_type="work",
    )
    assert [result.id for result in dated_results] == ["today"]
    assert dated_results[0].dense_score == pytest.approx(1.0)

    added = mgr.save("I cooked pasta yesterday")
    assert mgr._vector_matrix.shape == (3, 384)
    assert all(memory.vector is None for memory in mgr.list_memories())
    saved_records = json.loads(storage.read_text(encoding="utf-8"))["memories"]
    assert all(len(record["vector"]) == 384 for record in saved_records)
    pasta_vector = next(record["vector"] for record in saved_records if record["id"] == added.id)
    assert pasta_vector == pytest.approx(vector_pasta.tolist())

    assert mgr.delete("yesterday")
    assert mgr._vector_matrix.shape == (2, 384)
    assert added.id in {memory.id for memory in mgr.list_memories()}
    reloaded = MemoryManager(storage_path=storage, encoder_fn=mock_encoder)
    assert reloaded._vector_matrix.shape == (2, 384)

    # Saving with embeddings disabled must preserve existing vectors, while a
    # changed text must not inherit the old text's embedding.
    disabled = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    disabled.save("I prefer quiet mornings")
    assert disabled._vector_matrix.shape == (3, 384)
    persisted = json.loads(storage.read_text(encoding="utf-8"))["memories"]
    vectors_by_id = {record["id"]: record["vector"] for record in persisted}
    assert vectors_by_id["today"] == pytest.approx(vector_today.tolist())
    assert vectors_by_id[added.id] == pytest.approx(vector_pasta.tolist())

    assert disabled.update("today", "I wrote notes about a new bicycle")
    persisted = json.loads(storage.read_text(encoding="utf-8"))["memories"]
    vectors_by_id = {record["id"]: record["vector"] for record in persisted}
    assert vectors_by_id["today"] is None
    assert vectors_by_id[added.id] == pytest.approx(vector_pasta.tolist())

    refreshed = MemoryManager(storage_path=storage, encoder_fn=mock_encoder)
    assert refreshed._vector_matrix.shape == (3, 384)
    assert all(refreshed._vector_valid)


def test_memory_update_preserves_original_anchor_for_same_relative_event_phrase(tmp_path):
    mgr = MemoryManager(storage_path=tmp_path / "anchored.json", embedder=MemoryEmbedder(disabled=True))
    record = mgr.save("Last year, I stole a car")
    record.created_at = "2000-06-01T12:00:00+00:00"

    assert mgr.update(record.id, "Last year, I stole a car after midnight")
    updated = mgr.list_memories()[0]
    assert updated.event_date_start == "1999-01-01"
    assert updated.event_date_end == "1999-12-31"


def test_memory_manager_loads_legacy_format(tmp_path):
    legacy_file = tmp_path / "legacy_memories.json"
    legacy_data = {
        "version": 1,
        "memories": [
            {
                "id": "abc12345",
                "text": "every single time OBS opens, you must select the window or the display.",
                "created_at": "2026-09-30T00:00:00Z"
            }
        ]
    }
    legacy_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    mgr = MemoryManager(storage_path=legacy_file, embedder=MemoryEmbedder(disabled=True))
    assert len(mgr.list_memories()) == 1
    mem = mgr.list_memories()[0]
    assert mem.id == "abc12345"
    assert "OBS opens" in mem.text

    # Search on legacy memory
    results = mgr.search("how should I set up OBS?")
    assert len(results) >= 1
    assert results[0].id == "abc12345"


def test_memory_rejects_irrelevant_queries(tmp_path):
    storage = tmp_path / "test_reject.json"
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    mgr.save("My home wifi network is FiberOne and password is SafeKey99")
    mgr.save("My car license plate is 7XYZ999")

    # Irrelevant queries should return empty list
    assert mgr.search("what is the weather like in Tokyo?") == []
    assert mgr.search("calculate 25 times 40") == []
    assert mgr.search("turn off the living room lights") == []


def test_bm25_subset_scores_match_full_index():
    index = BM25Index()
    index.fit(["a", "b", "c"], ["worked a late shift", "made pasta", "worked the morning shift"])
    full = index.score("work shift")
    subset = index.score_subset("work shift", [0, 2])
    assert subset == {0: full[0], 2: full[2]}


def test_sparse_bm25_scores_match_dense_api_and_omit_nonmatches():
    index = BM25Index()
    index.fit(
        ["a", "b", "c"],
        ["worked a late shift", "made pasta", "worked the morning shift"],
    )
    dense = index.score("work shift")
    sparse = index.score_sparse("work shift")
    assert sparse == {idx: score for idx, score in enumerate(dense) if score > 0}
    assert 1 not in sparse
    assert index.score_sparse("words absent from store") == {}


def test_bm25_selects_dense_path_for_common_terms():
    index = BM25Index()
    index.fit(
        [str(i) for i in range(100)],
        [f"project {i}" for i in range(100)],
    )
    assert index.is_sparse_query("zircon")
    assert not index.is_sparse_query("project zircon")


def test_lexical_memory_search_uses_sparse_scores_without_dense_store_arrays(tmp_path):
    mgr = MemoryManager(
        storage_path=tmp_path / "sparse.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    expected = mgr.save("I worked on project zircon from 3:45 to 8:00pm")
    mgr.save("I cooked pasta with tomatoes and basil")
    mgr.bm25.score = lambda _query: (_ for _ in ()).throw(
        AssertionError("lexical search should not allocate full-store BM25 scores")
    )

    results = mgr.search("What time did I work on project zircon?")
    assert [result.id for result in results] == [expected.id]


def test_temporal_event_parser_keeps_precision_and_original_phrase():
    last_year = parse_event_time("Last year, I stole a car", "2026-10-03T09:30:00-04:00")
    assert last_year is not None
    assert last_year.date_start == "2025-01-01"
    assert last_year.date_end == "2025-12-31"
    assert last_year.precision == "year"
    assert last_year.original_expression == "last year"

    shift = parse_event_time(
        "I worked today from 3:45 to 8:00pm",
        "2026-10-03T09:30:00-04:00",
    )
    assert shift is not None
    assert shift.event_type == "work"
    assert shift.date_start == "2026-10-03"
    assert shift.start_at == "2026-10-03T15:45:00-04:00"
    assert shift.end_at == "2026-10-03T20:00:00-04:00"
    assert "3:45 to 8:00pm" in shift.original_expression

    dotted_shift = parse_event_time(
        "I worked today from 3.45 to 8 p.m.",
        "2026-10-03T09:30:00-04:00",
    )
    assert dotted_shift is not None
    assert dotted_shift.start_at == "2026-10-03T15:45:00-04:00"
    assert dotted_shift.end_at == "2026-10-03T20:00:00-04:00"

    day_before = parse_event_time("Day before yesterday I went hiking", "2026-10-03T09:30:00-04:00")
    assert day_before is not None
    assert day_before.date_start == "2026-10-01"


def test_temporal_event_clock_offsets_follow_dst_and_preserve_unresolved_times():
    spring_shift = parse_event_time(
        "I worked on 2026-03-08 from 1:30am to 3:30am",
        "2026-10-05T12:00:00-04:00",
        timezone_name="America/New_York",
    )
    assert spring_shift is not None
    assert spring_shift.start_at == "2026-03-08T01:30:00-05:00"
    assert spring_shift.end_at == "2026-03-08T03:30:00-04:00"
    assert spring_shift.timezone == "America/New_York"

    spring_gap = parse_event_time(
        "I worked on 2026-03-08 from 1:30am to 2:30am",
        "2026-10-05T12:00:00-04:00",
        timezone_name="America/New_York",
    )
    assert spring_gap is not None
    assert spring_gap.start_at is None
    assert spring_gap.end_at is None
    assert "1:30am to 2:30am" in spring_gap.original_expression

    fall_overlap = parse_event_time(
        "I worked on 2026-11-01 from 1:30am to 2:30am",
        "2026-10-05T12:00:00-04:00",
        timezone_name="America/New_York",
    )
    assert fall_overlap is not None
    assert fall_overlap.start_at is None
    assert fall_overlap.end_at is None
    assert "1:30am to 2:30am" in fall_overlap.original_expression

    overnight = parse_event_time(
        "I worked on 2026-03-07 from 11:00pm to 3:00am",
        "2026-10-05T12:00:00-04:00",
        timezone_name="America/New_York",
    )
    assert overnight is not None
    assert overnight.date_end == "2026-03-08"
    assert overnight.start_at == "2026-03-07T23:00:00-05:00"
    assert overnight.end_at == "2026-03-08T03:00:00-04:00"


def test_temporal_relative_day_uses_capture_timezone_and_memory_retains_it(monkeypatch, tmp_path):
    shifted_capture = parse_event_time(
        "Today, I worked from 3:45 to 8:00pm",
        "2026-03-09T02:30:00-04:00",
        timezone_name="America/Los_Angeles",
    )
    assert shifted_capture is not None
    assert shifted_capture.date_start == "2026-03-08"
    assert shifted_capture.start_at == "2026-03-08T15:45:00-07:00"
    assert shifted_capture.timezone == "America/Los_Angeles"

    monkeypatch.setattr("src.memory.manager.local_timezone_name", lambda: "America/Los_Angeles")
    mgr = MemoryManager(storage_path=tmp_path / "timezone-memory.json", embedder=MemoryEmbedder(disabled=True))
    saved = mgr.save("I worked today from 3:45 to 8:00pm")
    assert saved.event_timezone == "America/Los_Angeles"
    assert saved.event_start_at.endswith("-07:00")

    assert mgr.update(saved.id, "I worked today from 3:45 to 9:00pm")
    updated = mgr.list_memories()[0]
    assert updated.event_timezone == "America/Los_Angeles"
    assert updated.event_end_at.endswith("-07:00")


def test_temporal_memory_search_is_date_and_type_bounded_and_persists(tmp_path):
    storage = tmp_path / "dated_memories.json"
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    current = mgr.save("I worked today from 3:45 to 8:00pm")
    old = mgr.save("I worked 1 year ago from 9:00am to 5:00pm")
    mgr.save("I cooked pasta today")

    assert current.event_start_at is not None
    assert old.event_date_start is not None
    results = mgr.search(
        "What times did I work this week?",
        limit=20,
        date_range=(
            (datetime.now().astimezone().date() - timedelta(days=datetime.now().astimezone().weekday())).isoformat(),
            (datetime.now().astimezone().date() + timedelta(days=6 - datetime.now().astimezone().weekday())).isoformat(),
        ),
        event_type="work",
    )
    assert [result.id for result in results] == [current.id]

    context = mgr.retrieve_context("What times did I work this week?")
    assert current.text in context
    assert current.event_start_at in context
    assert old.text not in context

    reloaded = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    loaded = next(memory for memory in reloaded.list_memories() if memory.id == current.id)
    assert loaded.event_start_at == current.event_start_at
    assert loaded.event_time_expression == current.event_time_expression


def test_temporal_memory_context_includes_iana_timezone(monkeypatch, tmp_path):
    monkeypatch.setattr("src.memory.manager.local_timezone_name", lambda: "America/New_York")
    mgr = MemoryManager(storage_path=tmp_path / "timezone-context.json", embedder=MemoryEmbedder(disabled=True))
    saved = mgr.save("I worked on 2026-09-29 from 9:30 to 10:15am")

    context = mgr.retrieve_context("What timezone did I use for the 2026-09-29 work entry?")

    assert saved.event_timezone == "America/New_York"
    assert saved.event_start_at == "2026-09-29T09:30:00-04:00"
    assert context is not None
    assert "[event timezone: America/New_York]" in context


def test_relative_temporal_context_exposes_resolved_window_and_timezone(monkeypatch, tmp_path):
    from datetime import datetime as RealDateTime
    from zoneinfo import ZoneInfo

    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            fixed = cls(2026, 10, 6, 12, tzinfo=ZoneInfo("America/New_York"))
            return fixed.astimezone(tz) if tz is not None else fixed

        def astimezone(self, tz=None):
            # The temporal parser asks for the host-local time with no explicit zone.
            return self if tz is None else super().astimezone(tz)

    monkeypatch.setattr("src.memory.manager.local_timezone_name", lambda: "America/New_York")
    monkeypatch.setattr("src.memory.temporal.datetime", FixedDateTime)
    monkeypatch.setenv("TZ", "America/New_York")
    mgr = MemoryManager(
        storage_path=tmp_path / "relative-window-context.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    in_window_first = mgr.save(
        "I worked on the Juniper migration on 2026-09-29 from 9:30 to 10:15am"
    )
    in_window_second = mgr.save(
        "I worked on the Juniper invoice export on 2026-10-02 from 3:00pm to 4:00pm"
    )
    outside_window = mgr.save(
        "I worked on Juniper cleanup on 2026-10-05 from 11:00am to 11:30am"
    )

    relative_context = mgr.retrieve_context("What Juniper work did I log last week?")
    assert relative_context is not None
    assert relative_context.splitlines()[0] == (
        "[Resolved relative date window: 2026-09-28 through 2026-10-04; "
        "local timezone: America/New_York]"
    )
    assert relative_context.index(in_window_first.text) < relative_context.index(
        in_window_second.text
    )
    assert in_window_first.event_start_at in relative_context
    assert in_window_second.event_start_at in relative_context
    assert outside_window.text not in relative_context

    explicit_context = mgr.retrieve_context("What Juniper work did I log on 2026-09-29?")
    assert explicit_context is not None
    assert not explicit_context.startswith("[Resolved relative date window:")
    assert in_window_first.text in explicit_context
    assert in_window_second.text not in explicit_context
    assert outside_window.text not in explicit_context

    mixed_context = mgr.retrieve_context(
        "What Juniper work did I log on 2026-09-29 last week?"
    )
    assert mixed_context is not None
    assert not mixed_context.startswith("[Resolved relative date window:")
    assert in_window_first.text in mixed_context
    assert in_window_second.text not in mixed_context

    non_temporal_context = mgr.retrieve_context("Juniper migration")
    assert non_temporal_context is not None
    assert not non_temporal_context.startswith("[Resolved relative date window:")
    assert in_window_first.text in non_temporal_context

    assert mgr.retrieve_context("What Juniper work did I log last year?") is None

    # An unsupported process-local TZ must not inherit the system-zone fallback
    # returned by local_timezone_name(), since it may not match datetime's anchor.
    monkeypatch.setenv("TZ", "UTC0")
    override_context = mgr.retrieve_context("What Juniper work did I log last week?")
    assert override_context is not None
    assert override_context.splitlines()[0] == (
        "[Resolved relative date window: 2026-09-28 through 2026-10-04; "
        "local timezone: unavailable]"
    )

    monkeypatch.delenv("TZ", raising=False)
    unset_context = mgr.retrieve_context("What Juniper work did I log last week?")
    assert unset_context is not None
    assert unset_context.splitlines()[0].endswith("local timezone: America/New_York]")


@pytest.mark.asyncio
async def test_brain_recall_request_receives_synthetic_event_timezone(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from src.llm.brain import AdamBrain

    monkeypatch.setattr("src.memory.manager.local_timezone_name", lambda: "America/New_York")
    memory = MemoryManager(
        storage_path=tmp_path / "synthetic-brain-memory.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    memory.save("I worked on the Juniper migration on 2026-09-29 from 9:30 to 10:15am")

    config = SimpleNamespace(
        computer_control=SimpleNamespace(enabled=False, ocr_only=True),
        computer_vision=SimpleNamespace(enabled=False),
        llm=SimpleNamespace(
            provider="custom",
            local_model="test-model",
            cloud_model="test-model",
            ollama_host="http://localhost:11434",
            api_base="https://example.test/v1",
            api_key="test-key",
            temperature=0,
            num_ctx=8192,
            max_tool_rounds=2,
        ),
    )

    class RecallClient:
        provider = "custom"

        def __init__(self):
            self.requests = []

        async def chat(self, messages, tools=None, max_tokens=None, think=None):
            self.requests.append(messages)
            return {"content": "The recorded timezone was America/New_York.", "tool_calls": []}

    client = RecallClient()
    tts = SimpleNamespace(speak_async=AsyncMock())
    brain = AdamBrain(config, None, None, None, tts, memory_mgr=memory)
    brain.llm_client = client

    await brain.process_user_utterance("What timezone did I use for the 2026-09-29 work entry?")

    assert len(client.requests) == 1
    user_message = next(message for message in client.requests[0] if message.get("role") == "user")
    assert "I worked on the Juniper migration" in user_message["content"]
    assert "2026-09-29T09:30:00-04:00" in user_message["content"]
    assert "[event timezone: America/New_York]" in user_message["content"]
    assert tts.speak_async.await_args.args[0] == "The recorded timezone was America/New_York."


def test_memory_search_rejects_single_token_false_positive_in_large_store(tmp_path):
    mgr = MemoryManager(storage_path=tmp_path / "specific.json", embedder=MemoryEmbedder(disabled=True))
    expected = mgr.save("I worked on project zircon from 3:45 to 8:00pm")
    mgr.save("I worked on project maple from 8:00am to 12:30pm")
    mgr.save("I worked on project cobalt from 1:15 to 6:00pm")

    results = mgr.search("What time did I work on project zircon?", limit=3)

    assert [result.id for result in results] == [expected.id]


def test_legacy_memories_are_not_assigned_guessed_event_dates(tmp_path):
    storage = tmp_path / "old.json"
    storage.write_text(json.dumps({"version": 1, "memories": [{
        "id": "legacy", "text": "I worked from 3:45 to 8:00pm", "created_at": "2020-01-01T00:00:00Z"
    }]}), encoding="utf-8")
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    legacy = mgr.list_memories()[0]
    assert legacy.event_date_start is None
    assert mgr.retrieve_context("What times did I work this week?") is None


def test_invalid_event_date_metadata_is_excluded_from_temporal_index(tmp_path):
    storage = tmp_path / "malformed.json"
    storage.write_text(json.dumps({"memories": [{
        "id": "bad-date", "text": "I worked today", "event_type": "work",
        "event_date_start": "2026-99-99", "event_date_end": "2026-99-99"
    }]}), encoding="utf-8")
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))
    assert mgr.search("worked this week", date_range=("2026-10-01", "2026-10-07"), event_type="work") == []


def test_current_memory_query_prefers_explicit_correction_but_history_keeps_both(tmp_path):
    mgr = MemoryManager(
        storage_path=tmp_path / "preference-corrections.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    mgr.save("I prefer dark mode in my code editor.", category="preferences")
    mgr.save("Correction: I now prefer light mode in my code editor.", category="preferences")
    mgr.save("I prefer sepia mode in my photo editor.", category="preferences")

    current_query = "What theme do I currently prefer in my code editor?"
    current_context = mgr.retrieve_context(current_query)
    assert current_context is not None
    assert current_context.splitlines()[0] == (
        "- Correction: I now prefer light mode in my code editor."
    )
    assert "I prefer dark mode in my code editor." in current_context
    assert "I prefer sepia mode in my photo editor." not in current_context

    from src.llm.brain import _is_memory_only_recall_request

    assert _is_memory_only_recall_request(current_query, current_context)
    assert _is_memory_only_recall_request(
        "What is my latest code editor theme preference?", current_context
    )
    assert not _is_memory_only_recall_request(
        "What is my current CPU usage?", current_context
    )
    assert not _is_memory_only_recall_request(
        "When is my current appointment?", current_context
    )

    history_context = mgr.retrieve_context(
        "Did I change my code editor from dark to light mode?"
    )
    assert history_context is not None
    assert history_context.splitlines() == [
        "- I prefer dark mode in my code editor.",
        "- Correction: I now prefer light mode in my code editor.",
    ]


@pytest.mark.asyncio
async def test_brain_current_preference_recall_uses_corrected_memory_without_tools(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.llm.brain import AdamBrain, MEMORY_RECALL_SYSTEM_PROMPT

    memory = MemoryManager(
        storage_path=tmp_path / "brain-preference-corrections.json",
        embedder=MemoryEmbedder(disabled=True),
    )
    memory.save("I prefer dark mode in my code editor.", category="preferences")
    memory.save("Correction: I now prefer light mode in my code editor.", category="preferences")

    class RecallClient:
        provider = "custom"

        async def chat(self, messages, tools=None, **_kwargs):
            self.messages = messages
            self.tools = tools
            return {
                "content": "You currently prefer light mode in your code editor.",
                "tool_calls": [],
            }

    config = SimpleNamespace(
        computer_control=SimpleNamespace(enabled=False, ocr_only=True),
        computer_vision=SimpleNamespace(enabled=False),
        llm=SimpleNamespace(
            provider="custom", local_model="test-model", cloud_model="test-model",
            ollama_host="http://localhost:11434", api_base="https://example.test/v1",
            api_key="test-key", temperature=0, num_ctx=8192, max_tool_rounds=2,
        ),
    )
    tts = SimpleNamespace(speak_async=AsyncMock())
    brain = AdamBrain(config, None, None, None, tts, memory_mgr=memory)
    client = RecallClient()
    brain.llm_client = client
    brain.skill_manager.get_matched_skill_context = lambda _query: None

    await brain.process_user_utterance("What theme do I currently prefer in my editor?")

    assert client.tools == []
    assert client.messages[0]["content"] == MEMORY_RECALL_SYSTEM_PROMPT
    user_message = next(
        message for message in client.messages
        if message.get("role") == "user"
    )
    correction_position = user_message["content"].index(
        "Correction: I now prefer light mode"
    )
    prior_position = user_message["content"].index(
        "I prefer dark mode in my code editor."
    )
    assert correction_position < prior_position
    assert "prefer a later explicit correction or update" in client.messages[0]["content"]


def test_similar_dated_events_are_not_semantically_merged(tmp_path):
    def same_vector(texts):
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = 1.0
        return np.tile(vec, (len(texts), 1))

    mgr = MemoryManager(storage_path=tmp_path / "events.json", encoder_fn=same_vector)
    first = mgr.save("I worked today from 8:00am to 3:00pm")
    second = mgr.save("I worked today from 3:45 to 8:00pm")
    assert first.id != second.id
    assert len(mgr.list_memories()) == 2


@pytest.mark.asyncio
async def test_brain_manage_memory_tool(tmp_path):
    storage = tmp_path / "brain_memories.json"
    mgr = MemoryManager(storage_path=storage, embedder=MemoryEmbedder(disabled=True))

    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    config = SimpleNamespace(
        computer_control=SimpleNamespace(enabled=False, ocr_only=True),
        computer_vision=SimpleNamespace(enabled=False),
        llm=SimpleNamespace(
            provider="custom",
            local_model="test-model",
            cloud_model="test-model",
            ollama_host="http://localhost:11434",
            api_base="https://example.test/v1",
            api_key="test-key",
            temperature=0,
            num_ctx=8192,
            max_tool_rounds=2,
        ),
    )
    supervisor = SimpleNamespace()
    probe = SimpleNamespace()
    confirmation = SimpleNamespace()
    tts = SimpleNamespace(speak_async=AsyncMock())

    from src.llm.brain import AdamBrain
    brain = AdamBrain(
        config=config,
        supervisor=supervisor,
        probe=probe,
        confirmation_mgr=confirmation,
        tts_engine=tts,
        memory_mgr=mgr,
    )

    # 1. Test tool execution: save
    res = await brain._execute_tool("manage_memory", {"action": "save", "text": "I like dark theme in all IDEs", "category": "preference"})
    data = json.loads(res)
    assert data["ok"] is True
    mem_id = data["id"]
    assert len(mgr.list_memories()) == 1

    # 2. Test tool execution: search
    res = await brain._execute_tool("manage_memory", {"action": "search", "query": "what theme do I like?"})
    data = json.loads(res)
    assert data["ok"] is True
    assert data["count"] >= 1
    assert "dark theme" in data["matches"][0]["text"]

    # 3. Test tool execution: list
    res = await brain._execute_tool("manage_memory", {"action": "list"})
    data = json.loads(res)
    assert data["ok"] is True
    assert len(data["memories"]) == 1

    # 4. Test tool execution: update
    res = await brain._execute_tool("manage_memory", {"action": "update", "memory_id": mem_id, "text": "I prefer light theme now"})
    data = json.loads(res)
    assert data["ok"] is True
    assert mgr.list_memories()[0].text == "I prefer light theme now"

    # 5. Test tool execution: delete
    res = await brain._execute_tool("manage_memory", {"action": "delete", "memory_id": mem_id})
    data = json.loads(res)
    assert data["ok"] is True
    assert len(mgr.list_memories()) == 0
