"""Lightweight, lazy embedding wrapper for dense semantic matching.

Designed for minimal computational overhead on portable devices:
- Reuses existing IdeaRouter encoder if already loaded in RAM.
- Caches query embeddings with an LRU cache to avoid redundant inferences.
- Gracefully falls back to None if model or dependencies are unavailable,
  allowing BM25 lexical search to proceed with zero errors.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Embedding vector dimension for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM = 384


class MemoryEmbedder:
    """Manages lightweight query and document embeddings."""

    def __init__(
        self,
        encoder_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
        model_dir: str | Path | None = None,
        disabled: bool = False,
    ) -> None:
        self._encoder_fn = encoder_fn
        self._model_dir = Path(model_dir) if model_dir else None
        self._disabled = disabled
        self._session = None
        self._tokenizer = None
        self._load_failed = False

    @property
    def is_available(self) -> bool:
        """True if an encoder function is set or local ONNX model can be loaded."""
        if self._disabled or self._load_failed:
            return False
        if self._encoder_fn is not None:
            return True
        try:
            self._ensure_loaded()
            return self._session is not None
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        if self._encoder_fn is not None or self._session is not None or self._disabled or self._load_failed:
            return

        try:
            from src.intent.idea_router import (
                MODEL_CACHE,
                MODEL_ID,
                download_model,
                selected_onnx_file,
            )
            from tokenizers import Tokenizer
            import onnxruntime as ort

            root = self._model_dir or MODEL_CACHE
            onnx_name = selected_onnx_file()
            model_path = root / onnx_name
            tok_path = root / "tokenizer.json"

            if not model_path.is_file() or not tok_path.is_file():
                # Attempt to download model if missing
                try:
                    root = download_model(MODEL_ID, cache_dir=root)
                    model_path = root / onnx_name
                    tok_path = root / "tokenizer.json"
                except Exception:
                    self._load_failed = True
                    return

            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._tokenizer.enable_truncation(max_length=256)
            self._tokenizer.enable_padding(length=256)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            self._load_failed = True
            # Silent fallback: memory system works via BM25

    def encode(self, texts: Sequence[str]) -> np.ndarray | None:
        """Encode a batch of texts into normalized 384-dimensional vectors."""
        if not texts or self._disabled or self._load_failed:
            return None

        # 1. Use injected encoder if available (e.g. from IdeaRouter)
        if self._encoder_fn is not None:
            try:
                vecs = self._encoder_fn(texts)
                if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
                    return vecs.astype(np.float32)
            except Exception:
                pass

        # 2. Lazy load own lightweight ONNX session
        self._ensure_loaded()
        if self._session is None or self._tokenizer is None:
            return None

        try:
            encoded = self._tokenizer.encode_batch(list(texts))
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
            )
            # Mean pooling over attention mask
            token_embeddings = outputs[0]  # shape: (batch_size, seq_len, 384)
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 Normalize
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            return (mean_pooled / norms).astype(np.float32)
        except Exception:
            return None

    @functools.lru_cache(maxsize=64)
    def encode_query(self, query: str) -> tuple[float, ...] | None:
        """Encode a single query string with LRU caching.
        
        Returns a tuple of floats (for hashability in lru_cache) or None.
        """
        vecs = self.encode([query])
        if vecs is not None and len(vecs) > 0:
            return tuple(float(x) for x in vecs[0])
        return None
