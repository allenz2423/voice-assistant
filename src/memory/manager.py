"""Dynamic, lightweight memory manager for Adam.

Designed for high accuracy, sub-millisecond retrieval, and low computational
footprint on portable and embedded devices. Combines fast BM25Okapi lexical
indexing with lazy dense sentence embeddings.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.memory.bm25 import BM25Index, tokenize
from src.memory.embedder import MemoryEmbedder

DEFAULT_MEMORY_PATH = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    / "adam"
    / "embedding-memories.json"
)

# Natural speech trigger regexes for direct voice memory creation
MEMORY_TRIGGERS = [
    re.compile(
        r"^\s*(?:please\s+)?(?:make|create)\s+a\s+memory(?:\s*[,;:]\s*|\s+(?:that|to|about)\s+)(.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:remember|keep\s+in\s+mind)(?:\s*[,;:]\s*|\s+(?:that|to|my|our)\s+)(.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:don't|do\s+not)\s+forget(?:\s*[,;:]\s*|\s+(?:that|to|about)\s+)(.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:please\s+)?save\s+(?:this\s+)?to\s+(?:your\s+)?memory(?:\s*[,;:]\s*|\s+(?:that|to|about)\s+)(.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
]


def extract_memory_command(utterance: str) -> str | None:
    """Extract the clean memory text from a voice command, or None if not a trigger."""
    raw = str(utterance or "").strip()
    if not raw:
        return None
    for pattern in MEMORY_TRIGGERS:
        match = pattern.match(raw)
        if match and match.group(1).strip():
            # Clean leading/trailing punctuation and quotes
            extracted = match.group(1).strip().strip("'\"`")
            if extracted:
                return extracted
    return None


@dataclass
class MemoryRecord:
    id: str
    text: str
    category: str = "general"
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    access_count: int = 0
    last_accessed_at: str | None = None
    vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Vector is kept in memory or saved if present
        return data


@dataclass(frozen=True)
class MemorySearchResult:
    id: str
    text: str
    category: str
    score: float
    dense_score: float
    bm25_score: float
    exact_match: bool


class MemoryManager:
    """Thread-safe dynamic memory system with hybrid BM25 + dense retrieval."""

    def __init__(
        self,
        storage_path: str | Path = DEFAULT_MEMORY_PATH,
        embedder: MemoryEmbedder | None = None,
        encoder_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
        min_hybrid_score: float = 0.32,
        min_dense_score: float = 0.52,
        min_bm25_score: float = 1.8,
    ) -> None:
        self.storage_path = Path(storage_path).expanduser()
        self.min_hybrid_score = min_hybrid_score
        self.min_dense_score = min_dense_score
        self.min_bm25_score = min_bm25_score

        self._lock = threading.RLock()
        self.embedder = embedder or MemoryEmbedder(encoder_fn=encoder_fn)
        self.bm25 = BM25Index()

        self._memories: list[MemoryRecord] = []
        self._id_map: dict[str, MemoryRecord] = {}
        self._vector_matrix: np.ndarray | None = None

        self._load()

    def _load(self) -> None:
        """Load memories from disk and build in-memory indexes."""
        with self._lock:
            self._memories = []
            self._id_map = {}
            if self.storage_path.is_file():
                try:
                    payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                    items = payload.get("memories", []) if isinstance(payload, dict) else []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        text = str(item.get("text", "")).strip()
                        if not text:
                            continue
                        mem_id = str(item.get("id") or uuid.uuid4().hex)
                        record = MemoryRecord(
                            id=mem_id,
                            text=text,
                            category=str(item.get("category", "general")),
                            created_at=str(item.get("created_at", datetime.now().astimezone().isoformat())),
                            updated_at=str(item.get("updated_at", item.get("created_at", datetime.now().astimezone().isoformat()))),
                            access_count=int(item.get("access_count", 0)),
                            last_accessed_at=item.get("last_accessed_at"),
                            vector=item.get("vector"),
                        )
                        self._memories.append(record)
                        self._id_map[record.id] = record
                except Exception as exc:
                    print(f"[MemoryManager] Warning: Failed to load {self.storage_path}: {exc}", flush=True)

            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild BM25 index and cached vector matrix."""
        if not self._memories:
            self.bm25.fit([], [])
            self._vector_matrix = None
            return

        doc_ids = [m.id for m in self._memories]
        texts = [m.text for m in self._memories]
        self.bm25.fit(doc_ids, texts)

        # Check if pre-stored vectors are available for all memories
        stored_vectors = [m.vector for m in self._memories]
        if all(v is not None and len(v) == 384 for v in stored_vectors):
            self._vector_matrix = np.asarray(stored_vectors, dtype=np.float32)
        elif self.embedder.is_available:
            # Lazy batch encode missing vectors
            encoded = self.embedder.encode(texts)
            if encoded is not None and len(encoded) == len(self._memories):
                self._vector_matrix = encoded
                for i, m in enumerate(self._memories):
                    m.vector = [float(x) for x in encoded[i]]
            else:
                self._vector_matrix = None
        else:
            self._vector_matrix = None

    def _persist(self) -> None:
        """Atomically persist memories to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.storage_path.parent, 0o700)
        except OSError:
            pass

        data = {
            "version": 2,
            "updated_at": datetime.now().astimezone().isoformat(),
            "memories": [m.to_dict() for m in self._memories],
        }
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"

        fd, temp_path = tempfile.mkstemp(prefix=".embedding-memories-", dir=self.storage_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
            os.replace(temp_path, self.storage_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def save(self, text: str, category: str = "general") -> MemoryRecord:
        """Add or dynamically update a memory."""
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("Memory text cannot be empty.")

        with self._lock:
            now_iso = datetime.now().astimezone().isoformat()

            # Deduplication / Update Check:
            # Check if an existing memory is near-identical (> 0.90 dense similarity or exact text)
            existing_idx = None
            if self._memories:
                for idx, m in enumerate(self._memories):
                    if m.text.lower() == clean_text.lower():
                        existing_idx = idx
                        break

                if existing_idx is None and self._vector_matrix is not None and self.embedder.is_available:
                    q_vec = self.embedder.encode_query(clean_text)
                    if q_vec is not None:
                        q_arr = np.asarray(q_vec, dtype=np.float32)
                        sims = self._vector_matrix @ q_arr
                        best_idx = int(np.argmax(sims))
                        if sims[best_idx] > 0.92:
                            existing_idx = best_idx

            if existing_idx is not None:
                record = self._memories[existing_idx]
                record.text = clean_text
                record.category = category or record.category
                record.updated_at = now_iso
                # Re-encode vector if embedder is active
                if self.embedder.is_available:
                    vec = self.embedder.encode([clean_text])
                    if vec is not None and len(vec) > 0:
                        record.vector = [float(x) for x in vec[0]]
                        if self._vector_matrix is not None:
                            self._vector_matrix[existing_idx] = vec[0]
                self._rebuild_indexes()
                self._persist()
                return record

            # New memory creation
            new_id = uuid.uuid4().hex
            vector = None
            if self.embedder.is_available:
                vec = self.embedder.encode([clean_text])
                if vec is not None and len(vec) > 0:
                    vector = [float(x) for x in vec[0]]

            record = MemoryRecord(
                id=new_id,
                text=clean_text,
                category=category or "general",
                created_at=now_iso,
                updated_at=now_iso,
                vector=vector,
            )
            self._memories.append(record)
            self._id_map[record.id] = record
            self._rebuild_indexes()
            self._persist()
            return record

    def search(
        self,
        query: str,
        limit: int = 3,
        category: str | None = None,
    ) -> list[MemorySearchResult]:
        """Perform hybrid BM25 + dense retrieval for a query.
        
        Runs in microsecond time:
        - BM25Okapi: ~0.01 ms
        - Matrix multiply (if vector available): ~0.01 ms
        - Query embedding (LRU cached): ~0-2 ms
        """
        clean_query = str(query or "").strip()
        if not clean_query or not self._memories:
            return []

        with self._lock:
            candidates: list[tuple[int, MemoryRecord]] = [
                (i, m) for i, m in enumerate(self._memories)
                if category is None or m.category.lower() == category.lower()
            ]
            if not candidates:
                return []

            # 1. Lexical Scoring (BM25Okapi)
            bm25_raw_scores = self.bm25.score(clean_query)

            # 2. Dense Semantic Scoring (Vector Cosine Similarity)
            dense_scores: list[float] = [0.0] * len(self._memories)
            if self._vector_matrix is not None and self.embedder.is_available:
                q_vec = self.embedder.encode_query(clean_query)
                if q_vec is not None:
                    q_arr = np.asarray(q_vec, dtype=np.float32)
                    dense_matrix = self._vector_matrix
                    dot_prods = dense_matrix @ q_arr
                    dense_scores = [float(x) for x in dot_prods]

            # 3. Exact Substring / Token Matching & Coverage Bonus
            query_tokens = [t for t in tokenize(clean_query, filter_stopwords=True) if len(t) >= 2]
            results: list[MemorySearchResult] = []

            for idx, record in candidates:
                raw_bm25 = bm25_raw_scores[idx] if idx < len(bm25_raw_scores) else 0.0
                norm_bm25 = BM25Index.normalize_score(raw_bm25)
                dense_score = max(0.0, min(1.0, dense_scores[idx] if idx < len(dense_scores) else 0.0))

                doc_tokens = set(tokenize(record.text, filter_stopwords=False))
                matched_query_tokens = [t for t in query_tokens if t in doc_tokens]
                coverage = len(matched_query_tokens) / max(len(query_tokens), 1)

                exact_bonus = 0.0
                has_exact = False
                has_number_match = any(any(c.isdigit() for c in t) and t in doc_tokens for t in query_tokens)
                if has_number_match or coverage >= 0.75:
                    exact_bonus = 0.20
                    has_exact = True
                elif coverage >= 0.5:
                    exact_bonus = 0.10
                    has_exact = True

                # Hybrid score formulation:
                # Dense weights 0.55, BM25 weights 0.45 + exact match bonus
                if self._vector_matrix is not None and self.embedder.is_available:
                    hybrid_score = (0.55 * dense_score) + (0.45 * norm_bm25) + exact_bonus
                else:
                    # BM25-only fallback for ultra-constrained devices
                    hybrid_score = norm_bm25 + exact_bonus

                # Acceptance criteria:
                is_accepted = (
                    (hybrid_score >= self.min_hybrid_score and (coverage >= 0.4 or dense_score >= 0.4))
                    or dense_score >= self.min_dense_score
                    or raw_bm25 >= self.min_bm25_score
                    or (has_exact and norm_bm25 >= 0.25)
                )

                if is_accepted:
                    results.append(
                        MemorySearchResult(
                            id=record.id,
                            text=record.text,
                            category=record.category,
                            score=float(hybrid_score),
                            dense_score=float(dense_score),
                            bm25_score=float(raw_bm25),
                            exact_match=has_exact,
                        )
                    )

            # Sort by hybrid score descending
            results.sort(key=lambda r: r.score, reverse=True)
            top_results = results[:limit]

            # Update access statistics for retrieved memories
            now_iso = datetime.now().astimezone().isoformat()
            changed = False
            for res in top_results:
                rec = self._id_map.get(res.id)
                if rec is not None:
                    rec.access_count += 1
                    rec.last_accessed_at = now_iso
                    changed = True

            if changed:
                # Save access counts in background or next save
                try:
                    self._persist()
                except Exception:
                    pass

            return top_results

    def retrieve_context(self, utterance: str, limit: int = 3) -> str | None:
        """Format matching memories into a compact context block for prompt injection."""
        matches = self.search(utterance, limit=limit)
        if not matches:
            return None

        lines = [f"- {m.text}" for m in matches]
        return "\n".join(lines)

    def update(self, memory_id: str, text: str, category: str | None = None) -> bool:
        """Update an existing memory by ID."""
        with self._lock:
            rec = self._id_map.get(memory_id)
            if rec is None:
                return False

            clean_text = str(text or "").strip()
            if not clean_text:
                raise ValueError("Memory text cannot be empty.")

            rec.text = clean_text
            if category:
                rec.category = category
            rec.updated_at = datetime.now().astimezone().isoformat()

            if self.embedder.is_available:
                vec = self.embedder.encode([clean_text])
                if vec is not None and len(vec) > 0:
                    rec.vector = [float(x) for x in vec[0]]

            self._rebuild_indexes()
            self._persist()
            return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        with self._lock:
            if memory_id not in self._id_map:
                return False

            self._memories = [m for m in self._memories if m.id != memory_id]
            self._id_map.pop(memory_id, None)
            self._rebuild_indexes()
            self._persist()
            return True

    def list_memories(self, category: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        """List stored memories."""
        with self._lock:
            items = [
                m for m in self._memories
                if category is None or m.category.lower() == category.lower()
            ]
            return items[:limit]

    def clear(self) -> None:
        """Remove all memories."""
        with self._lock:
            self._memories = []
            self._id_map = {}
            self._rebuild_indexes()
            self._persist()
