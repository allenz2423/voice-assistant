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
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.memory.bm25 import BM25Index, tokenize
from src.memory.embedder import MemoryEmbedder
from src.memory.temporal import (
    event_type_for_query,
    local_timezone_name,
    parse_event_time,
    parse_temporal_query_range,
)

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


def _event_time_fields(
    text: str,
    recorded_at: str,
    timezone_name: str | None = None,
) -> dict[str, str | None]:
    event = parse_event_time(text, recorded_at, timezone_name=timezone_name)
    return {
        "event_type": event.event_type if event else None,
        "event_date_start": event.date_start if event else None,
        "event_date_end": event.date_end if event else None,
        "event_start_at": event.start_at if event else None,
        "event_end_at": event.end_at if event else None,
        "event_time_expression": event.original_expression if event else None,
        "event_timezone": event.timezone if event else None,
        "event_precision": event.precision if event else None,
    }


def _valid_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


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
    event_type: str | None = None
    event_date_start: str | None = None
    event_date_end: str | None = None
    event_start_at: str | None = None
    event_end_at: str | None = None
    event_time_expression: str | None = None
    event_timezone: str | None = None
    event_precision: str | None = None

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
    event_type: str | None = None
    event_date_start: str | None = None
    event_date_end: str | None = None
    event_start_at: str | None = None
    event_end_at: str | None = None
    event_time_expression: str | None = None
    event_timezone: str | None = None
    event_precision: str | None = None


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
        self._vector_valid: list[bool] = []
        self._event_end_index: list[tuple[str, int]] = []

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
                            event_type=item.get("event_type"),
                            event_date_start=item.get("event_date_start"),
                            event_date_end=item.get("event_date_end"),
                            event_start_at=item.get("event_start_at"),
                            event_end_at=item.get("event_end_at"),
                            event_time_expression=item.get("event_time_expression"),
                            event_timezone=item.get("event_timezone"),
                            event_precision=item.get("event_precision"),
                        )
                        self._memories.append(record)
                        self._id_map[record.id] = record
                except Exception as exc:
                    print(f"[MemoryManager] Warning: Failed to load {self.storage_path}: {exc}", flush=True)

            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild BM25 index and cached vector matrix."""
        previous_vector_matrix = self._vector_matrix
        previous_vector_valid = list(self._vector_valid)
        self._event_end_index = sorted(
            (record.event_date_end or record.event_date_start, idx)
            for idx, record in enumerate(self._memories)
            if _valid_iso_date(record.event_date_start)
            and _valid_iso_date(record.event_date_end or record.event_date_start)
            and record.event_date_start <= (record.event_date_end or record.event_date_start)
        )
        if not self._memories:
            self.bm25.fit([], [])
            self._vector_matrix = None
            self._vector_valid = []
            return

        doc_ids = [m.id for m in self._memories]
        texts = [m.text for m in self._memories]
        self.bm25.fit(doc_ids, texts)

        # Keep one compact float32 matrix in RAM instead of also retaining
        # 384 Python float objects on every MemoryRecord. The record vectors
        # are reconstructed only while serializing to the existing JSON format.
        stored_vectors = [m.vector for m in self._memories]
        if all(v is not None and len(v) == 384 for v in stored_vectors):
            self._vector_matrix = np.asarray(stored_vectors, dtype=np.float32)
            self._vector_valid = [True] * len(self._memories)
        elif previous_vector_matrix is not None and previous_vector_matrix.shape[0] == len(self._memories):
            self._vector_matrix = previous_vector_matrix
            self._vector_valid = (
                previous_vector_valid
                if len(previous_vector_valid) == len(self._memories)
                else [True] * len(self._memories)
            )
            for index, vector in enumerate(stored_vectors):
                if vector is not None and len(vector) == 384:
                    self._vector_matrix[index] = np.asarray(vector, dtype=np.float32)
                    self._vector_valid[index] = True
        elif (
            previous_vector_matrix is not None
            and previous_vector_matrix.shape[0] + 1 == len(self._memories)
        ):
            new_vector = stored_vectors[-1]
            has_new_vector = new_vector is not None and len(new_vector) == 384
            new_row = (
                np.asarray(new_vector, dtype=np.float32).reshape(1, -1)
                if has_new_vector
                else np.zeros((1, 384), dtype=np.float32)
            )
            self._vector_matrix = np.concatenate((previous_vector_matrix, new_row), axis=0)
            self._vector_valid = previous_vector_valid + [has_new_vector]
        elif self.embedder.is_available:
            # Lazy batch encode missing vectors
            encoded = self.embedder.encode(texts)
            if encoded is not None and len(encoded) == len(self._memories):
                self._vector_matrix = encoded
                self._vector_valid = [True] * len(self._memories)
            else:
                self._vector_matrix = None
                self._vector_valid = [False] * len(self._memories)
        elif any(vector is not None and len(vector) == 384 for vector in stored_vectors):
            # Preserve usable stored vectors even if another record has no
            # embedding and the embedder is currently unavailable.
            self._vector_matrix = np.zeros((len(self._memories), 384), dtype=np.float32)
            self._vector_valid = []
            for index, vector in enumerate(stored_vectors):
                valid = vector is not None and len(vector) == 384
                self._vector_valid.append(valid)
                if valid:
                    self._vector_matrix[index] = np.asarray(vector, dtype=np.float32)
        else:
            self._vector_matrix = None
            self._vector_valid = [False] * len(self._memories)

        if self._vector_matrix is not None:
            for memory in self._memories:
                memory.vector = None

    def _persist(self) -> None:
        """Atomically persist memories to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.storage_path.parent, 0o700)
        except OSError:
            pass

        serialized_memories = []
        for index, memory in enumerate(self._memories):
            item = memory.to_dict()
            if (
                self._vector_matrix is not None
                and self._vector_matrix.shape[0] == len(self._memories)
            ):
                item["vector"] = (
                    self._vector_matrix[index].tolist()
                    if index < len(self._vector_valid) and self._vector_valid[index]
                    else None
                )
            serialized_memories.append(item)

        data = {
            "version": 2,
            "updated_at": datetime.now().astimezone().isoformat(),
            "memories": serialized_memories,
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
            temporal_fields = _event_time_fields(clean_text, now_iso, local_timezone_name())
            has_event_date = temporal_fields["event_date_start"] is not None

            # Deduplication / Update Check:
            # Check if an existing memory is near-identical (> 0.90 dense similarity or exact text)
            existing_idx = None
            if self._memories:
                for idx, m in enumerate(self._memories):
                    same_event = (
                        m.event_date_start == temporal_fields["event_date_start"]
                        and m.event_type == temporal_fields["event_type"]
                    )
                    same_kind = (m.event_date_start is not None) == has_event_date
                    if m.text.lower() == clean_text.lower() and (same_event if has_event_date else same_kind):
                        existing_idx = idx
                        break

                if (
                    not has_event_date
                    and existing_idx is None
                    and self._vector_matrix is not None
                    and self.embedder.is_available
                ):
                    q_vec = self.embedder.encode_query(clean_text)
                    if q_vec is not None:
                        q_arr = np.asarray(q_vec, dtype=np.float32)
                        sims = self._vector_matrix @ q_arr
                        eligible = [
                            idx for idx, memory in enumerate(self._memories)
                            if memory.event_date_start is None
                        ]
                        best_idx = max(eligible, key=lambda idx: sims[idx], default=None)
                        if best_idx is not None and sims[best_idx] > 0.92:
                            existing_idx = best_idx

            if existing_idx is not None:
                record = self._memories[existing_idx]
                text_changed = record.text != clean_text
                record.text = clean_text
                record.category = category or record.category
                record.updated_at = now_iso
                for field_name, value in temporal_fields.items():
                    setattr(record, field_name, value)
                # Re-encode vector if embedder is active
                if self.embedder.is_available:
                    vec = self.embedder.encode([clean_text])
                    if vec is not None and len(vec) > 0:
                        record.vector = [float(x) for x in vec[0]]
                        if self._vector_matrix is not None:
                            self._vector_matrix[existing_idx] = vec[0]
                    elif text_changed:
                        record.vector = None
                elif text_changed:
                    record.vector = None
                if text_changed and record.vector is None and self._vector_matrix is not None:
                    self._vector_matrix[existing_idx].fill(0.0)
                    self._vector_valid[existing_idx] = False
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
                **temporal_fields,
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
        date_range: tuple[str, str] | None = None,
        event_type: str | None = None,
    ) -> list[MemorySearchResult]:
        """Perform hybrid retrieval, using sparse lexical candidates when selective.

        Lexical-only lookups avoid scoring and allocating data for unrelated
        records when query postings are sparse. Broad queries keep the dense
        BM25 path, and dated lookups prefilter through the temporal index.
        """
        clean_query = str(query or "").strip()
        if not clean_query or not self._memories:
            return []

        with self._lock:
            lexical_only = not (
                self._vector_matrix is not None and self.embedder.is_available
            )
            sparse_bm25_query = self.bm25.is_sparse_query(clean_query)
            sparse_lexical_query = bool(
                lexical_only
                and date_range is None
                and event_type is None
                and sparse_bm25_query
            )
            if date_range is not None:
                range_start, range_end = date_range
                first = bisect_right(self._event_end_index, (range_start, -1))
                candidate_indices = [
                    idx for _event_end, idx in self._event_end_index[first:]
                    if (self._memories[idx].event_date_start or "") <= range_end
                ]
            elif sparse_lexical_query:
                # With no dense semantic channel, documents absent from the
                # lexical postings cannot pass any later acceptance rule.
                candidate_indices = sorted(self.bm25.score_sparse(clean_query))
            else:
                candidate_indices = list(range(len(self._memories)))
            candidates: list[tuple[int, MemoryRecord]] = [
                (i, self._memories[i]) for i in candidate_indices
                if (category is None or self._memories[i].category.lower() == category.lower())
                and (event_type is None or self._memories[i].event_type == event_type)
            ]
            if not candidates:
                return []

            # 1. Lexical Scoring (BM25Okapi)
            if date_range is not None or event_type is not None:
                bm25_raw_scores = self.bm25.score_subset(clean_query, [idx for idx, _ in candidates])
            elif sparse_bm25_query:
                bm25_raw_scores = self.bm25.score_sparse(clean_query)
            else:
                bm25_raw_scores = self.bm25.score(clean_query)

            # 2. Dense Semantic Scoring (Vector Cosine Similarity)
            # Keep full-store scores in a compact NumPy array. For filtered
            # queries, retain scores only for candidates instead of allocating
            # a Python list sized to every stored memory.
            dense_scores: np.ndarray | dict[int, float] | None = None
            if not lexical_only:
                q_vec = self.embedder.encode_query(clean_query)
                if q_vec is not None:
                    q_arr = np.asarray(q_vec, dtype=np.float32)
                    indices = [idx for idx, _ in candidates]
                    if len(indices) == len(self._memories):
                        # Fancy indexing would copy the entire N x 384 matrix.
                        dense_scores = self._vector_matrix @ q_arr
                    else:
                        dot_prods = self._vector_matrix[indices] @ q_arr
                        dense_scores = {idx: float(score) for idx, score in zip(indices, dot_prods)}

            # 3. Exact Substring / Token Matching & Coverage Bonus
            query_tokens = [t for t in tokenize(clean_query, filter_stopwords=True) if len(t) >= 2]
            results: list[MemorySearchResult] = []

            for idx, record in candidates:
                raw_bm25 = (
                    bm25_raw_scores.get(idx, 0.0)
                    if isinstance(bm25_raw_scores, dict)
                    else bm25_raw_scores[idx] if idx < len(bm25_raw_scores) else 0.0
                )
                norm_bm25 = BM25Index.normalize_score(raw_bm25)
                if isinstance(dense_scores, dict):
                    raw_dense_score = dense_scores.get(idx, 0.0)
                elif dense_scores is not None and idx < len(dense_scores):
                    raw_dense_score = float(dense_scores[idx])
                else:
                    raw_dense_score = 0.0
                dense_score = max(0.0, min(1.0, raw_dense_score))

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
                if not lexical_only:
                    hybrid_score = (0.55 * dense_score) + (0.45 * norm_bm25) + exact_bonus
                else:
                    # BM25-only fallback for ultra-constrained devices
                    hybrid_score = norm_bm25 + exact_bonus

                # Acceptance criteria:
                is_accepted = (
                    (hybrid_score >= self.min_hybrid_score and (coverage >= 0.4 or dense_score >= 0.4))
                    or dense_score >= self.min_dense_score
                    # A single rare shared token can exceed the BM25 score
                    # threshold in a large store. Keep lexical threshold
                    # matches grounded in enough of the query to avoid
                    # injecting loosely related memories into the prompt.
                    or (raw_bm25 >= self.min_bm25_score and coverage >= 0.4)
                    or (has_exact and norm_bm25 >= 0.25)
                    or date_range is not None
                    or event_type is not None
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
                            event_type=record.event_type,
                            event_date_start=record.event_date_start,
                            event_date_end=record.event_date_end,
                            event_start_at=record.event_start_at,
                            event_end_at=record.event_end_at,
                            event_time_expression=record.event_time_expression,
                            event_timezone=record.event_timezone,
                            event_precision=record.event_precision,
                        )
                    )

            # Sort by hybrid score descending
            results.sort(key=lambda r: r.score, reverse=True)
            top_results = results[:limit]

            # Update access statistics for retrieved memories
            now_iso = datetime.now().astimezone().isoformat()
            for res in top_results:
                rec = self._id_map.get(res.id)
                if rec is not None:
                    rec.access_count += 1
                    rec.last_accessed_at = now_iso
            # Read-side statistics stay in memory. Rewriting the entire JSON
            # store for every lookup made successful retrieval scale with the
            # size of the file; the next explicit memory mutation persists them.

            return top_results

    def retrieve_context(self, utterance: str, limit: int = 3) -> str | None:
        """Format matching memories into a compact context block for prompt injection."""
        date_range = parse_temporal_query_range(utterance)
        event_type = event_type_for_query(utterance)
        matches = self.search(
            utterance,
            limit=max(limit, 20) if date_range is not None else limit,
            date_range=date_range,
            event_type=event_type if date_range is not None else None,
        )
        if not matches:
            return None

        if date_range is not None:
            matches.sort(key=lambda item: item.event_start_at or item.event_date_start or "")

        lines = []
        for memory in matches:
            event_note = ""
            if memory.event_start_at and memory.event_end_at:
                event_note = f" [resolved event time: {memory.event_start_at} to {memory.event_end_at}]"
            elif memory.event_date_start:
                if memory.event_date_start == memory.event_date_end:
                    event_note = f" [resolved event date: {memory.event_date_start}]"
                else:
                    event_note = (
                        f" [resolved event date range: {memory.event_date_start} through "
                        f"{memory.event_date_end}]"
                    )
                if memory.event_precision:
                    event_note += f" [date precision: {memory.event_precision}]"
                if memory.event_time_expression:
                    event_note += f" [original time phrase: {memory.event_time_expression}]"
            if memory.event_start_at and memory.event_time_expression:
                event_note += f" [original time phrase: {memory.event_time_expression}]"
            if memory.event_timezone:
                event_note += f" [event timezone: {memory.event_timezone}]"
            lines.append(f"- {memory.text}{event_note}")
        return "\n".join(lines)

    def update(self, memory_id: str, text: str, category: str | None = None) -> bool:
        """Update an existing memory by ID."""
        with self._lock:
            rec = self._id_map.get(memory_id)
            if rec is None:
                return False
            memory_index = next(
                index for index, memory in enumerate(self._memories)
                if memory.id == memory_id
            )

            clean_text = str(text or "").strip()
            if not clean_text:
                raise ValueError("Memory text cannot be empty.")

            text_changed = rec.text != clean_text
            rec.text = clean_text
            if category:
                rec.category = category
            rec.updated_at = datetime.now().astimezone().isoformat()
            old_relative_phrase = (rec.event_time_expression or "").split(";", 1)[0].strip()
            keep_original_time_anchor = bool(
                old_relative_phrase
                and re.search(re.escape(old_relative_phrase), clean_text, re.IGNORECASE)
            )
            event_anchor = rec.created_at if keep_original_time_anchor else rec.updated_at
            event_timezone = (
                rec.event_timezone if keep_original_time_anchor else local_timezone_name()
            )
            for field_name, value in _event_time_fields(
                clean_text, event_anchor, event_timezone
            ).items():
                setattr(rec, field_name, value)

            if self.embedder.is_available:
                vec = self.embedder.encode([clean_text])
                if vec is not None and len(vec) > 0:
                    rec.vector = [float(x) for x in vec[0]]
                elif text_changed:
                    rec.vector = None
            elif text_changed:
                rec.vector = None

            if text_changed and rec.vector is None and self._vector_matrix is not None:
                self._vector_matrix[memory_index].fill(0.0)
                self._vector_valid[memory_index] = False

            self._rebuild_indexes()
            self._persist()
            return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        with self._lock:
            memory_index = next(
                (index for index, memory in enumerate(self._memories) if memory.id == memory_id),
                None,
            )
            if memory_index is None:
                return False

            if (
                self._vector_matrix is not None
                and self._vector_matrix.shape[0] == len(self._memories)
            ):
                self._vector_matrix = np.delete(self._vector_matrix, memory_index, axis=0)
                if len(self._vector_valid) == len(self._memories):
                    self._vector_valid = [
                        valid for index, valid in enumerate(self._vector_valid)
                        if index != memory_index
                    ]
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
