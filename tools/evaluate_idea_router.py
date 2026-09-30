"""Evaluate MiniLM idea retrieval against the checked-in 2,000-phrase corpus."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intent.idea_router import IdeaRouter

PHRASE_FILE = ROOT / "tests" / "data" / "idea_router_phrases_2000.jsonl"


def main() -> None:
    router = IdeaRouter()
    ideas = router._load_ideas()
    rows = [json.loads(line) for line in PHRASE_FILE.read_text(encoding="utf-8").splitlines()]
    # Single-item batches mirror production queries and avoid changing scores
    # through different quantized ONNX padding shapes.
    queries = router.encode([row["text"] for row in rows], batch_size=1)
    scores = queries @ router._get_idea_vectors().T
    winners = np.argmax(scores, axis=1)
    top_scores = scores[np.arange(len(rows)), winners]
    second_scores = np.partition(scores, -2, axis=1)[:, -2]
    margins = top_scores - second_scores
    predicted_routes = np.asarray([ideas[index]["route"] for index in winners])
    expected = np.asarray([row["expected"] for row in rows])
    expected_routes = np.where(
        expected == "direct_action",
        "command",
        np.where(expected == "personal_calendar_commitment", "background_calendar", "ignore"),
    )

    print(f"Model: {router.model_id}")
    print(f"Phrases: {len(rows)}; thresholds: {router.thresholds}; margin: {router.minimum_margin:.3f}")
    for route in ("command", "background_calendar"):
        accepted = (
            (predicted_routes == route)
            & (top_scores >= router.thresholds[route])
            & (margins >= router.minimum_margin)
        )
        truth = expected_routes == route
        true_positive = int(np.sum(accepted & truth))
        false_positive = int(np.sum(accepted & ~truth))
        false_negative = int(np.sum(~accepted & truth))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        print(
            f"{route}: accepted={int(accepted.sum())}, TP={true_positive}, FP={false_positive}, "
            f"FN={false_negative}, precision={precision:.1%}, recall={recall:.1%}"
        )

    print("\nRepresentative decisions:")
    for phrase in (
        "Open Spotify",
        "I have a meeting tomorrow at 3:30 pm.",
        "I had a meeting yesterday.",
        "Maybe I should schedule a meeting next week.",
        "My sister has a doctor appointment tomorrow.",
        "The video says someone has a meeting tomorrow.",
    ):
        match = router.match(phrase)
        print(f"  {phrase} -> {match}")


if __name__ == "__main__":
    main()
