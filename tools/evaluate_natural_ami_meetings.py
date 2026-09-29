"""Evaluate Adam on naturally occurring AMI meetings with original mix audio.

The selected EN2002 recordings are real AMI Mix-Headset WAVs. Reference turns
come from the aligned per-speaker IHM annotations. Run with the ``eval`` and
``nemotron-diarization`` dependency groups enabled.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_meeting_ami import edit_distance, map_speakers, normalize_words
from src.config import load_config
from src.stt.diarizer import NemotronDiarizer
from src.stt.transcriber import create_transcriber


MIX_DATASET = "diarizers-community/ami"
REFERENCE_DATASET = "edinburghcstr/ami"
MEETINGS = {
    # 25 second spans with all participants speaking and available timed text.
    "EN2002b": 900.0,
    "EN2002c": 890.0,
}
MIX_SHARDS = 3
REFERENCE_SHARD = "ihm/test-00000-of-00004.parquet"


@dataclass
class NaturalClipResult:
    meeting_id: str
    start: float
    duration: float
    participants: int
    reference_words: int
    full_mix_baseline_errors: int
    overall_errors: int
    speaker_reference_words: int
    speaker_errors: int
    mapping: dict[str, str]
    turns: list[dict]


def read_mix_audio(meeting_ids: set[str]) -> dict[str, bytes]:
    import pyarrow.parquet as pq

    found: dict[str, bytes] = {}
    for shard_index in range(MIX_SHARDS):
        shard_name = f"ihm/test-{shard_index:05d}-of-{MIX_SHARDS:05d}.parquet"
        path = hf_hub_download(MIX_DATASET, shard_name, repo_type="dataset")
        table = pq.read_table(path, columns=["audio"])
        for item in table.column("audio").to_pylist():
            path_name = item.get("path") or ""
            meeting_id = Path(path_name).name.split(".", 1)[0]
            if meeting_id in meeting_ids and item.get("bytes"):
                found[meeting_id] = item["bytes"]
        del table
        if meeting_ids <= found.keys():
            break
    missing = meeting_ids - found.keys()
    if missing:
        raise RuntimeError(f"Original AMI mix audio not found for: {', '.join(sorted(missing))}")
    return found


def read_references(meeting_ids: set[str]) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    path = hf_hub_download(REFERENCE_DATASET, REFERENCE_SHARD, repo_type="dataset")
    table = pq.read_table(
        path,
        columns=["meeting_id", "speaker_id", "begin_time", "end_time", "text"],
    )
    result = {meeting_id: [] for meeting_id in meeting_ids}
    for row in table.to_pylist():
        if row["meeting_id"] in result:
            result[row["meeting_id"]].append(row)
    missing = [meeting_id for meeting_id, rows in result.items() if not rows]
    if missing:
        raise RuntimeError(
            "Timed transcript references were not found in the selected annotation shard for: "
            + ", ".join(missing)
        )
    return result


def evaluate_window(meeting_id: str, mix_bytes: bytes, annotations: list[dict],
                    start_seconds: float, duration_seconds: float,
                    diarizer: NemotronDiarizer, transcriber) -> NaturalClipResult:
    with sf.SoundFile(io.BytesIO(mix_bytes)) as mix:
        sample_rate = mix.samplerate
        if sample_rate != 16000 or mix.channels != 1:
            raise ValueError(f"{meeting_id} mix must be mono 16 kHz audio")
        mix.seek(int(start_seconds * sample_rate))
        audio = mix.read(int(duration_seconds * sample_rate), dtype="float32")
    clip_end = start_seconds + len(audio) / sample_rate

    references: dict[str, list[tuple[float, float, str]]] = {}
    for row in annotations:
        begin, end = float(row["begin_time"]), float(row["end_time"])
        if end <= start_seconds or begin >= clip_end:
            continue
        left, right = max(begin, start_seconds), min(end, clip_end)
        speaker = str(row["speaker_id"])
        references.setdefault(speaker, []).append(
            (left - start_seconds, right - start_seconds, str(row["text"] or ""))
        )
    references = {speaker: rows for speaker, rows in references.items() if rows}

    turns = diarizer.speaker_turns(audio, diarizer.diarize(audio), sample_rate=sample_rate)
    mapping = map_speakers(turns, references)
    predictions = []
    for turn in turns:
        text = transcriber.transcribe(turn.audio).strip()
        if text:
            predictions.append((turn.start, turn.speakers, text))

    reference_all = sorted(
        (start, speaker, text)
        for speaker, rows in references.items()
        for start, _, text in rows
    )
    reference_words = normalize_words(" ".join(text for _, _, text in reference_all))
    hypothesis_words = normalize_words(" ".join(text for _, _, text in sorted(predictions)))
    baseline_words = normalize_words(transcriber.transcribe(audio))

    speaker_errors = 0
    speaker_reference_words = 0
    for speaker, rows in references.items():
        reference = normalize_words(" ".join(text for _, _, text in rows))
        hypothesis = normalize_words(" ".join(
            text for _, ids, text in predictions
            if len(ids) == 1 and mapping.get(ids[0]) == speaker
        ))
        speaker_errors += edit_distance(reference, hypothesis)
        speaker_reference_words += len(reference)

    return NaturalClipResult(
        meeting_id=meeting_id,
        start=start_seconds,
        duration=len(audio) / sample_rate,
        participants=len(references),
        reference_words=len(reference_words),
        full_mix_baseline_errors=edit_distance(reference_words, baseline_words),
        overall_errors=edit_distance(reference_words, hypothesis_words),
        speaker_reference_words=speaker_reference_words,
        speaker_errors=speaker_errors,
        mapping=mapping,
        turns=[
            {"start": round(start, 2), "speakers": list(ids), "text": text}
            for start, ids, text in predictions
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--device", default="cpu", help="Nemotron device; defaults to CPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    diarizer = NemotronDiarizer(model="nvidia/Nemotron-3-Diarization", device=args.device)
    if not diarizer.available or not diarizer.supports_model():
        raise SystemExit("Download the optional Nemotron Transformers runtime and model first")

    ids = set(MEETINGS)
    print("Fetching original AMI meeting mixes and timed speaker transcripts...")
    mixes = read_mix_audio(ids)
    references = read_references(ids)
    transcriber = create_transcriber(config.stt, shared_api_key=config.llm.api_key)
    results = []
    for meeting_id, start in MEETINGS.items():
        result = evaluate_window(
            meeting_id,
            mixes[meeting_id],
            references[meeting_id],
            start,
            args.duration,
            diarizer,
            transcriber,
        )
        results.append(result)
        overall_wer = result.overall_errors / max(1, result.reference_words)
        baseline_wer = result.full_mix_baseline_errors / max(1, result.reference_words)
        speaker_wer = result.speaker_errors / max(1, result.speaker_reference_words)
        print(
            f"{meeting_id} {start:.0f}-{start + result.duration:.0f}s; "
            f"{result.participants} participants; turn WER={overall_wer:.1%}; "
            f"full-mix WER={baseline_wer:.1%}; "
            f"speaker WER={speaker_wer:.1%}; map={result.mapping}"
        )

    ref_words = sum(result.reference_words for result in results)
    ref_speaker_words = sum(result.speaker_reference_words for result in results)
    summary = {
        "mix_dataset": MIX_DATASET,
        "reference_dataset": REFERENCE_DATASET,
        "meetings": [asdict(result) for result in results],
        "clips": len(results),
        "total_audio_seconds": sum(result.duration for result in results),
        "full_mix_baseline_wer": sum(result.full_mix_baseline_errors for result in results) / max(1, ref_words),
        "overall_wer": sum(result.overall_errors for result in results) / max(1, ref_words),
        "speaker_attributed_wer": sum(result.speaker_errors for result in results) / max(1, ref_speaker_words),
        "speaker_wer_excludes_composite_overlap_turns": True,
    }
    print(
        f"Summary: {summary['clips']} natural meetings, {summary['total_audio_seconds']:.1f}s; "
        f"turn WER={summary['overall_wer']:.1%}; full-mix WER={summary['full_mix_baseline_wer']:.1%}; "
        f"speaker-attributed WER={summary['speaker_attributed_wer']:.1%} "
        "(shared overlap turns excluded from speaker WER)."
    )
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.output}")


if __name__ == "__main__":
    main()
