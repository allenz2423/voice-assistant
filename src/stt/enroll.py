"""Interactively enroll or replace Adam's local speaker-verification profile."""

from __future__ import annotations

import argparse
import gc
import sys

import numpy as np

from src.audio.stream import AudioStreamManager
from src.config import load_config
from src.stt.speaker import SAMPLE_RATE, SpeakerVerifier


PROMPTS = (
    "I use Adam to help me with things on my computer.",
    "Please check what is on my screen and tell me what you see.",
    "This is my voice profile, recorded locally for speaker verification.",
)


def record_sample(
    stream: AudioStreamManager,
    silence_duration: float = 1.25,
    max_duration: float = 15.0,
    idle_threshold: float = 0.25,
    speaking_threshold: float = 0.85,
) -> np.ndarray:
    """Capture a whole utterance, ending after silence instead of a fixed timer."""
    audio = stream.record_utterance(
        silence_duration=silence_duration,
        max_duration=max_duration,
        idle_threshold=idle_threshold,
        speaking_threshold=speaking_threshold,
    )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < int(1.5 * SAMPLE_RATE):
        raise ValueError("I didn't capture enough speech. Please try that phrase again.")
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Adam configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    verifier = SpeakerVerifier()
    if verifier.enrolled:
        answer = input(f"A speaker profile exists at {verifier.profile_path}. Replace it? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Enrollment cancelled; existing profile was left unchanged.")
            return

    print("Enrollment records three short samples from your configured microphone.")
    print("Raw audio is kept only in memory and discarded; only a local voice embedding is saved.")
    samples = []
    stream = AudioStreamManager(
        target_source=config.audio.target_source,
        sample_rate=SAMPLE_RATE,
        chunk_size=512,
    )
    stream.start()
    try:
        for index, prompt in enumerate(PROMPTS, 1):
            while True:
                input(f"\nSample {index}/{len(PROMPTS)}. Press Enter when ready, then say: {prompt} ")
                print("Listening—pause briefly when you finish. (15-second maximum)")
                try:
                    samples.append(record_sample(
                        stream,
                        silence_duration=config.audio.vad_silence_duration,
                        idle_threshold=config.audio.vad_threshold_idle,
                        speaking_threshold=config.audio.vad_threshold_speaking,
                    ))
                    break
                except ValueError as exc:
                    print(exc)
    finally:
        stream.stop()

    try:
        verifier.enroll(samples)
    except Exception as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        raise
    finally:
        samples.clear()
        gc.collect()

    print(f"Speaker profile enrolled locally at {verifier.profile_path}.")
    print("Restart Adam to enable speaker checks. To remove the profile, delete that file.")


if __name__ == "__main__":
    main()
