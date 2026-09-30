"""Local ONNX sentence embeddings for matching utterances to editable ideas.

This module only retrieves candidate ideas. The caller decides whether to wake
Adam's normal command loop or run the restricted background observation flow.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import platform
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = Path.home() / ".cache" / "adam" / "models" / "all-MiniLM-L6-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDEAS = REPO_ROOT / "assets" / "intent_ideas.json"
DEFAULT_MEMORIES = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "adam" / "embedding-memories.json"
MAX_TOKENS = 256
MEMORY_MATCH_THRESHOLD = 0.72
MEMORY_MATCH_MARGIN = 0.08


def extract_memory_text(utterance: str) -> str | None:
    """Return the verbatim transcript after the explicit memory command."""
    match = re.match(
        r"^\s*make a memory(?:\s*[,;:]\s*|\s+(?:that|to)\s+)(.+?)\s*$",
        str(utterance or ""),
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match and match.group(1).strip() else None


@dataclass(frozen=True)
class IdeaMatch:
    idea_id: str
    title: str
    route: str
    score: float
    runner_up_score: float
    margin: float
    accepted: bool


@dataclass(frozen=True)
class MemoryMatch:
    memory_id: str
    text: str
    score: float
    runner_up_score: float
    margin: float
    accepted: bool


def selected_onnx_file(machine: str | None = None, cpu_flags: set[str] | None = None) -> str:
    """Choose a compact quantized ONNX file when the host CPU supports it."""
    machine = (machine or platform.machine()).lower()
    if machine in {"aarch64", "arm64"}:
        return "onnx/model_qint8_arm64.onnx"
    if machine in {"x86_64", "amd64"}:
        if cpu_flags is None:
            try:
                cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower()
                cpu_flags = set(re.findall(r"(?m)^flags\s*:\s*(.*)$", cpuinfo)[0].split())
            except (OSError, IndexError):
                cpu_flags = set()
        if "avx2" in cpu_flags:
            return "onnx/model_quint8_avx2.onnx"
    return "onnx/model.onnx"


def download_model(model_id: str = MODEL_ID, cache_dir: str | Path = MODEL_CACHE) -> Path:
    """Download only the tokenizer and host-appropriate ONNX model."""
    from huggingface_hub import snapshot_download

    onnx_file = selected_onnx_file()
    root = Path(snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        allow_patterns=["config.json", "tokenizer.json", onnx_file],
    ))
    required = (root / "tokenizer.json", root / onnx_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Downloaded embedding model is missing: " + ", ".join(missing))
    return root


class IdeaRouter:
    """Encode utterances locally and retrieve the closest configured idea."""

    def __init__(
        self,
        ideas_path: str | Path = DEFAULT_IDEAS,
        model_id: str = MODEL_ID,
        model_dir: str | Path | None = None,
        command_threshold: float = 0.30,
        background_threshold: float = 0.49,
        minimum_margin: float = 0.025,
        memories_path: str | Path = DEFAULT_MEMORIES,
    ) -> None:
        self.ideas_path = Path(ideas_path).expanduser()
        if not self.ideas_path.is_absolute():
            self.ideas_path = REPO_ROOT / self.ideas_path
        self.model_id = model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.thresholds = {
            "command": float(command_threshold),
            "background_calendar": float(background_threshold),
            "ignore": float("-inf"),
        }
        self.minimum_margin = float(minimum_margin)
        self.memories_path = Path(memories_path).expanduser()
        self._tokenizer = None
        self._session = None
        self._ideas: list[dict[str, Any]] | None = None
        self._idea_vectors: np.ndarray | None = None
        self._memories: list[dict[str, Any]] | None = None
        self._memory_vectors: np.ndarray | None = None

    def _load_memories(self) -> list[dict[str, Any]]:
        if self._memories is None:
            try:
                payload = json.loads(self.memories_path.read_text(encoding="utf-8"))
                memories = payload.get("memories", []) if isinstance(payload, dict) else []
            except FileNotFoundError:
                memories = []
            if not isinstance(memories, list):
                raise ValueError(f"Invalid memory store: {self.memories_path}")
            self._memories = [
                item for item in memories
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
        return self._memories

    def add_memory(self, text: str) -> str:
        """Persist the transcript text exactly as supplied; return its stable ID."""
        exact_text = str(text)
        if not exact_text.strip():
            raise ValueError("Memory text cannot be empty.")
        memories = self._load_memories()
        memory = {
            "id": uuid.uuid4().hex,
            "text": exact_text,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        updated = [*memories, memory]
        self.memories_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.memories_path.parent, 0o700)
        payload = json.dumps({"version": 1, "memories": updated}, ensure_ascii=False, indent=2) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix=".embedding-memories-", dir=self.memories_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.memories_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        self._memories = updated
        self._memory_vectors = None
        return str(memory["id"])

    def match_memory(self, utterance: str) -> MemoryMatch | None:
        """Retrieve a high-confidence saved memory for the current utterance."""
        text = " ".join(str(utterance or "").split())
        memories = self._load_memories()
        if not text or not memories:
            return None
        if self._memory_vectors is None:
            self._memory_vectors = self.encode([str(item["text"]) for item in memories])
        query = self.encode([text])[0]
        scores = self._memory_vectors @ query
        ranked = np.argsort(scores)[::-1]
        best_idx = int(ranked[0])
        score = float(scores[best_idx])
        runner_up = float(scores[int(ranked[1])]) if len(ranked) > 1 else -1.0
        margin = score - runner_up
        best = memories[best_idx]
        return MemoryMatch(
            memory_id=str(best["id"]),
            text=str(best["text"]),
            score=score,
            runner_up_score=runner_up,
            margin=margin,
            accepted=score >= MEMORY_MATCH_THRESHOLD and margin >= MEMORY_MATCH_MARGIN,
        )

    @staticmethod
    def _idea_text(idea: dict[str, Any]) -> str:
        return str(idea.get("embedding_text") or f"{idea['title']}. {idea['description']}")

    def _load_ideas(self) -> list[dict[str, Any]]:
        if self._ideas is None:
            payload = json.loads(self.ideas_path.read_text(encoding="utf-8"))
            ideas = payload.get("ideas") if isinstance(payload, dict) else None
            if not isinstance(ideas, list) or not ideas:
                raise ValueError(f"No idea definitions found in {self.ideas_path}")
            seen: set[str] = set()
            for idea in ideas:
                if not isinstance(idea, dict):
                    raise ValueError("Each idea must be a JSON object")
                idea_id = str(idea.get("id", "")).strip()
                route = str(idea.get("route", "")).strip()
                if not idea_id or idea_id in seen:
                    raise ValueError(f"Idea IDs must be non-empty and unique: {idea_id!r}")
                if route not in self.thresholds:
                    raise ValueError(f"Unsupported route {route!r} for idea {idea_id!r}")
                if not str(idea.get("title", "")).strip() or not str(idea.get("description", "")).strip():
                    raise ValueError(f"Idea {idea_id!r} needs a title and description")
                seen.add(idea_id)
            self._ideas = ideas
        return self._ideas

    def _load_model(self) -> None:
        if self._session is not None:
            return
        from tokenizers import Tokenizer
        import onnxruntime as ort

        onnx_file = selected_onnx_file()
        if self.model_dir:
            root = self.model_dir
        else:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(MODEL_CACHE),
                allow_patterns=["config.json", "tokenizer.json", onnx_file],
                local_files_only=True,
            ))
        tokenizer_path = root / "tokenizer.json"
        model_path = root / onnx_file
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"Idea embedding model is missing. Run: uv run python -m src.intent.idea_router --download"
            )

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab = tokenizer.get_vocab()
        pad_id = vocab.get("[PAD]", 0)
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = tokenizer
        self._session = session

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return normalized mean-pooled vectors using the model's attention mask."""
        self._load_model()
        assert self._tokenizer is not None and self._session is not None
        outputs: list[np.ndarray] = []
        input_names = {item.name for item in self._session.get_inputs()}
        output_name = self._session.get_outputs()[0].name

        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            encoded = self._tokenizer.encode_batch([str(text) for text in batch])
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            feed: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = np.asarray(
                    [item.type_ids for item in encoded], dtype=np.int64
                )
            hidden = self._session.run([output_name], feed)[0]
            mask = attention_mask.astype(np.float32)[..., None]
            pooled = (hidden.astype(np.float32) * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            outputs.append(pooled / np.maximum(norms, 1e-12))
        if not outputs:
            return np.zeros((0, 384), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _get_idea_vectors(self) -> np.ndarray:
        if self._idea_vectors is None:
            ideas = self._load_ideas()
            self._idea_vectors = self.encode([self._idea_text(idea) for idea in ideas])
        return self._idea_vectors

    def prepare(self) -> None:
        """Load the local model and validate/embed the configured idea file."""
        self._get_idea_vectors()

    def match(self, utterance: str) -> IdeaMatch | None:
        """Return the closest idea with route threshold and runner-up margin."""
        text = " ".join(str(utterance or "").split())
        if not text:
            return None
        ideas = self._load_ideas()
        vectors = self._get_idea_vectors()
        query = self.encode([text])[0]
        scores = vectors @ query
        ranked = np.argsort(scores)[::-1]
        best_idx = int(ranked[0])
        best = ideas[best_idx]
        score = float(scores[best_idx])
        runner_up = float(scores[int(ranked[1])]) if len(ranked) > 1 else -1.0
        margin = score - runner_up
        threshold = self.thresholds[best["route"]]
        return IdeaMatch(
            idea_id=str(best["id"]),
            title=str(best["title"]),
            route=str(best["route"]),
            score=score,
            runner_up_score=runner_up,
            margin=margin,
            accepted=score >= threshold and margin >= self.minimum_margin,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/test Adam's local idea embedding model")
    parser.add_argument("--download", action="store_true", help="Download the compact MiniLM ONNX model")
    args = parser.parse_args()
    if args.download:
        path = download_model()
        print(f"Idea embedding model is ready: {path}")
        return
    parser.error("choose --download")


if __name__ == "__main__":
    main()
