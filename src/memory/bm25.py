"""Lightweight, computationally inexpensive BM25Okapi lexical index.

Designed for microsecond retrieval on resource-constrained and portable devices
(e.g. Raspberry Pi, low-power SBCs, laptops). Pure Python and NumPy, zero
external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

# Common English stopwords to ignore during lexical indexing
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-'_][a-zA-Z0-9]+)*")


def tokenize(text: str, filter_stopwords: bool = True) -> list[str]:
    """Extract lowercased alphanumeric tokens from text."""
    tokens = [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
    return tokens


class BM25Index:
    """Ultra-fast, in-memory BM25Okapi lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.doc_term_freqs: list[Counter[str]] = []
        self.inverted_index: dict[str, list[int]] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build the index from document IDs and texts."""
        self.doc_ids = list(doc_ids)
        self.num_docs = len(self.doc_ids)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.inverted_index = {}
        self.idf = {}

        if self.num_docs == 0:
            self.avg_doc_length = 0.0
            return

        total_length = 0
        df: Counter[str] = Counter()

        for idx, text in enumerate(texts):
            tokens = tokenize(text, filter_stopwords=True)
            # If all tokens were stopwords, keep raw tokens so short docs aren't empty
            if not tokens:
                tokens = tokenize(text, filter_stopwords=False)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf:
                df[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(idx)

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0.0

        # Calculate BM25 standard IDF with small-corpus floor so small memory stores
        # (N = 1 to 50) don't artificially diminish single-document matches.
        for term, freq in df.items():
            base_idf = math.log(((self.num_docs - freq + 0.5) / (freq + 0.5)) + 1.0)
            self.idf[term] = max(base_idf, 1.0)

    def score(self, query: str) -> list[float]:
        """Compute BM25 scores for all documents given a query string."""
        if self.num_docs == 0:
            return []

        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize(query, filter_stopwords=False)
        if not query_tokens:
            return [0.0] * self.num_docs

        scores = [0.0] * self.num_docs
        matched_counts = [0] * self.num_docs
        k1 = self.k1
        b = self.b
        avg_len = self.avg_doc_length or 1.0
        unique_query_terms = set(query_tokens)
        num_query_terms = len(unique_query_terms)

        for q_term in unique_query_terms:
            postings = self.inverted_index.get(q_term)
            if not postings:
                continue
            term_idf = self.idf.get(q_term, 1.0)
            for doc_idx in postings:
                tf = self.doc_term_freqs[doc_idx][q_term]
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
                scores[doc_idx] += term_idf * (numerator / denominator)
                matched_counts[doc_idx] += 1

        final_scores = [0.0] * self.num_docs
        for idx, raw_score in enumerate(scores):
            if raw_score <= 0 or matched_counts[idx] == 0:
                continue
            overlap_ratio = matched_counts[idx] / max(num_query_terms, 1)
            final_scores[idx] = raw_score * (0.4 + 0.6 * overlap_ratio)
        return final_scores

    def is_sparse_query(self, query: str, max_posting_fraction: float = 0.1) -> bool:
        """Whether posting lists are small enough to favor sparse scoring."""
        if self.num_docs == 0:
            return True
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize(query, filter_stopwords=False)
        if not query_tokens:
            return True
        posting_bound = sum(
            len(self.inverted_index.get(term, ())) for term in set(query_tokens)
        )
        threshold = max(32, int(self.num_docs * max(0.0, min(max_posting_fraction, 1.0))))
        return posting_bound <= threshold

    def score_sparse(self, query: str) -> dict[int, float]:
        """Score only documents matching a query term, avoiding store-sized arrays."""
        if self.num_docs == 0:
            return {}

        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize(query, filter_stopwords=False)
        if not query_tokens:
            return {}

        scores: dict[int, float] = {}
        matched_counts: dict[int, int] = {}
        k1 = self.k1
        b = self.b
        avg_len = self.avg_doc_length or 1.0
        unique_query_terms = set(query_tokens)
        num_query_terms = len(unique_query_terms)

        for q_term in unique_query_terms:
            if q_term not in self.inverted_index:
                continue

            term_idf = self.idf.get(q_term, 1.0)

            for doc_idx in self.inverted_index[q_term]:
                tf = self.doc_term_freqs[doc_idx][q_term]
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_idf * (numerator / denominator)
                matched_counts[doc_idx] = matched_counts.get(doc_idx, 0) + 1

        # Multiply by term recall/overlap ratio so a single incidental match
        # in a long query doesn't score higher than a doc matching most query terms
        final_scores: dict[int, float] = {}
        for idx, raw_score in scores.items():
            if raw_score <= 0:
                continue
            overlap_ratio = matched_counts[idx] / max(num_query_terms, 1)
            # Differentiate strong overlap from weak overlap
            final_scores[idx] = raw_score * (0.4 + 0.6 * overlap_ratio)

        return final_scores

    def score_subset(self, query: str, document_indices: Sequence[int]) -> dict[int, float]:
        """Score only selected documents, useful after date/category prefiltering."""
        selected = sorted({i for i in document_indices if 0 <= i < self.num_docs})
        if not selected:
            return {}
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize(query, filter_stopwords=False)
        if not query_tokens:
            return {i: 0.0 for i in selected}

        allowed = set(selected)
        raw_scores = {i: 0.0 for i in selected}
        matched_counts = {i: 0 for i in selected}
        avg_len = self.avg_doc_length or 1.0
        for term in set(query_tokens):
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 1.0)
            for idx in postings:
                if idx not in allowed:
                    continue
                tf = self.doc_term_freqs[idx][term]
                doc_len = self.doc_lengths[idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
                raw_scores[idx] += idf * (tf * (self.k1 + 1.0) / denominator)
                matched_counts[idx] += 1

        unique_terms = max(len(set(query_tokens)), 1)
        return {
            idx: raw * (0.4 + 0.6 * matched_counts[idx] / unique_terms) if raw > 0 else 0.0
            for idx, raw in raw_scores.items()
        }

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """Map raw unbounded BM25 score smoothly to [0, 1)."""
        if raw_score <= 0.0:
            return 0.0
        return float(raw_score / (raw_score + 1.2))
