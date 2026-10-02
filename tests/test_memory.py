"""Unit tests for the dynamic memory system."""

import json
import numpy as np
import pytest

from src.memory.bm25 import BM25Index, tokenize
from src.memory.embedder import MemoryEmbedder
from src.memory.manager import (
    MemoryManager,
    extract_memory_command,
)


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
