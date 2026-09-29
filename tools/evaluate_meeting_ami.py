"""Evaluate Adam's meeting transcription against timed AMI speaker references.

Run with ``uv sync --group eval --extra nemotron-diarization`` followed by
``uv run python tools/evaluate_meeting_ami.py``. ASR uses the configured model;
Nemotron defaults to CPU so this benchmark never guesses a GPU.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.stt.diarizer import NemotronDiarizer, SpeakerTurn
from src.stt.transcriber import create_transcriber


DATASET = "Trelis/ami-2speaker-test"
PARQUET_FILE = "data/train-00000-of-00001.parquet"
TIMED_TEXT = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>\s*(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)
WORDS = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass
class ScoredClip:
    index: int
    duration: float
    reference_words: int
    overall_errors: int
    speaker_reference_words: int
    speaker_errors: int
    speaker_mapping: dict[str, str]
    turns: list[dict]


def parse_timed_text(value: str) -> list[tuple[float, float, str]]:
    """Parse AMI's <|start|> utterance <|end|> reference format."""
    return [
        (float(start), float(end), text.strip())
        for start, text, end in TIMED_TEXT.findall(value or "")
        if float(end) > float(start) and text.strip()
    ]


def normalize_words(text: str) -> list[str]:
    return WORDS.findall((text or "").lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1]


def temporal_overlap(turns: list[SpeakerTurn], speaker_id: str,
                     reference: list[tuple[float, float, str]]) -> float:
    total = 0.0
    for turn in turns:
        if turn.speakers != (speaker_id,):
            continue
        for start, end, _ in reference:
            total += max(0.0, min(turn.end, end) - max(turn.start, start))
    return total


def map_speakers(turns: list[SpeakerTurn], references: dict[str, list[tuple[float, float, str]]]) -> dict[str, str]:
    """Find the best one-to-one local Nemotron ID to AMI speaker mapping."""
    ids = sorted({speaker for turn in turns for speaker in turn.speakers})
    if not ids:
        return {}
    gold_ids = list(references)
    weights = {
        (predicted, gold): temporal_overlap(turns, predicted, references[gold])
        for predicted in ids for gold in gold_ids
    }
    mapping_size = min(len(ids), len(gold_ids))
    if mapping_size == 0:
        return {}

    best_score = -1.0
    best_mapping: dict[str, str] = {}
    # Use timed annotations to find the best one-to-one mapping, including
    # multiple-participant natural meetings and any extra model IDs.
    for predicted_ids in itertools.combinations(ids, mapping_size):
        for gold_order in itertools.permutations(gold_ids, mapping_size):
            score = sum(weights[predicted, gold] for predicted, gold in zip(predicted_ids, gold_order))
            if score > best_score:
                best_score = score
                best_mapping = dict(zip(predicted_ids, gold_order))
    return best_mapping


def score_clip(index: int, row: dict, diarizer: NemotronDiarizer, transcriber) -> ScoredClip:
    audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if audio.ndim != 1 or sample_rate != 16000:
        raise ValueError(f"AMI clip {index} must be mono 16 kHz audio")

    references = {
        "speaker1": parse_timed_text(row["speaker1_target"]),
        "speaker2": parse_timed_text(row["speaker2_target"]),
    }
    turns = diarizer.speaker_turns(audio, diarizer.diarize(audio), sample_rate=sample_rate)
    mapping = map_speakers(turns, references)

    predictions: list[tuple[float, tuple[str, ...], str]] = []
    for turn in turns:
        text = transcriber.transcribe(turn.audio).strip()
        if text:
            predictions.append((turn.start, turn.speakers, text))

    reference_all = [
        (start, speaker, text)
        for speaker, utterances in references.items()
        for start, _, text in utterances
    ]
    reference_all.sort(key=lambda item: (item[0], item[1]))
    reference_words = normalize_words(" ".join(text for _, _, text in reference_all))
    hypothesis_words = normalize_words(" ".join(text for _, _, text in sorted(predictions)))

    # Composite overlap turns remain in the all-speaker ASR score, but are
    # omitted from speaker WER because a mixed crop has no trustworthy single
    # speaker assignment.
    speaker_errors = 0
    speaker_reference_words = 0
    for gold_speaker in references:
        gold_text = " ".join(text for _, _, text in references[gold_speaker])
        hyp_text = " ".join(
            text for _, speaker_ids, text in predictions
            if len(speaker_ids) == 1 and mapping.get(speaker_ids[0]) == gold_speaker
        )
        reference = normalize_words(gold_text)
        hypothesis = normalize_words(hyp_text)
        speaker_errors += edit_distance(reference, hypothesis)
        speaker_reference_words += len(reference)

    return ScoredClip(
        index=index,
        duration=len(audio) / sample_rate,
        reference_words=len(reference_words),
        overall_errors=edit_distance(reference_words, hypothesis_words),
        speaker_reference_words=speaker_reference_words,
        speaker_errors=speaker_errors,
        speaker_mapping=mapping,
        turns=[
            {"start": round(start, 2), "speakers": list(ids), "text": text}
            for start, ids, text in predictions
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Adam YAML config")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N clips (default: all 50)")
    parser.add_argument("--offset", type=int, default=0, help="Start at this dataset row")
    parser.add_argument("--device", default="cpu", help="Nemotron device: cpu or explicitly selected cuda[:index]")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    config = load_config(args.config)
    model = "nvidia/Nemotron-3-Diarization"
    diarizer = NemotronDiarizer(model=model, device=args.device)
    if not diarizer.available:
        raise SystemExit("Nemotron benchmark requires the optional Transformers diarization runtime")
    if not diarizer.supports_model():
        raise SystemExit(f"Nemotron model {model!r} is not cached; run the setup model download first")

    print(f"Loading {DATASET} and Adam ASR ({config.stt.model_size}, {config.stt.device})...")
    parquet = hf_hub_download(DATASET, PARQUET_FILE, repo_type="dataset")
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()
    stop = len(rows) if args.limit <= 0 else min(len(rows), args.offset + args.limit)
    selected = rows[args.offset:stop]
    if not selected:
        raise SystemExit("No dataset rows selected")
    transcriber = create_transcriber(config.stt, shared_api_key=config.llm.api_key)

    results = []
    for index, row in enumerate(selected, start=args.offset):
        result = score_clip(index, row, diarizer, transcriber)
        results.append(result)
        overall_rate = result.overall_errors / max(1, result.reference_words)
        speaker_rate = result.speaker_errors / max(1, result.speaker_reference_words)
        print(
            f"[{index + 1}/{stop}] {result.duration:.1f}s "
            f"overall WER={overall_rate:.1%} speaker WER={speaker_rate:.1%} "
            f"map={result.speaker_mapping}"
        )

    total_reference = sum(result.reference_words for result in results)
    total_errors = sum(result.overall_errors for result in results)
    total_speaker_reference = sum(result.speaker_reference_words for result in results)
    total_speaker_errors = sum(result.speaker_errors for result in results)
    detected_two = sum(len(set(result.speaker_mapping.values())) >= 2 for result in results)
    summary = {
        "dataset": DATASET,
        "clips": len(results),
        "total_audio_seconds": round(sum(result.duration for result in results), 2),
        "clips_with_two_mapped_speakers": detected_two,
        "overall_wer": total_errors / max(1, total_reference),
        "speaker_attributed_wer": total_speaker_errors / max(1, total_speaker_reference),
        "speaker_wer_excludes_composite_overlap_turns": True,
        "results": [asdict(result) for result in results],
    }
    print(
        f"\nSummary: {len(results)} clips, {summary['total_audio_seconds'] / 60:.1f} minutes; "
        f"two-speaker mapping {detected_two}/{len(results)}; "
        f"overall WER {summary['overall_wer']:.1%}; "
        f"speaker-attributed WER {summary['speaker_attributed_wer']:.1%} "
        "(composite overlap turns excluded)."
    )
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.output}")


if __name__ == "__main__":
    main()
