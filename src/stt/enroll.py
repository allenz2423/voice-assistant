"""Interactively add or replace Adam's local speaker-verification examples."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

from src.audio.stream import AudioStreamManager
from src.config import load_config
from src.stt.speaker import SAMPLE_RATE, SpeakerVerifier, default_profile_path, default_user_profile_path


PROMPTS = (
    "I use Adam to help me with things on my computer.",
    "Please check what is on my screen and tell me what you see.",
    "This is my voice profile, recorded locally for speaker verification.",
)
PROFILE_GUIDANCE = {
    "near": "Use your normal near-microphone speaking position.",
    "close": "Move closer to the microphone than the near profile.",
    "far": "Use the far-microphone position you want Adam to recognize.",
}


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
    parser.add_argument(
        "--user", default="You",
        help="Name of a user listed under speaker_verification.users in config.yaml (default: You)",
    )
    parser.add_argument(
        "--profile", default="default",
        help="Distance/acoustic profile configured for this user (for example near, close, far)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Replace the existing enrolled-speaker examples instead of adding to them",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    configured_user = next(
        (user for user in config.speaker_verification.users if user.name.casefold() == args.user.casefold()),
        None,
    )
    if configured_user is None and args.user.casefold() != "you":
        parser.error(
            f"user {args.user!r} is not configured; add it under speaker_verification.users in {args.config} first"
        )
    user_name = configured_user.name if configured_user else "You"
    profile_name = args.profile.strip()
    if configured_user and configured_user.profiles:
        configured_profile = next(
            (key for key in configured_user.profiles if key.casefold() == profile_name.casefold()),
            None,
        )
        if configured_profile is None:
            parser.error(
                f"profile {profile_name!r} is not configured for {user_name}; "
                f"choose one of: {', '.join(configured_user.profiles)}"
            )
        profile_name = configured_profile
        configured_path = configured_user.profiles[profile_name]
        target_profile = (
            Path(configured_path).expanduser()
            if configured_path
            else default_user_profile_path(user_name, profile_name)
        )
        profile_paths = {profile_name: target_profile}
    elif configured_user:
        if profile_name.casefold() != "default":
            parser.error(f"{user_name} has only the default profile; configure profiles under this user first")
        profile_name = "default"
        target_profile = (
            Path(configured_user.profile_path).expanduser()
            if configured_user.profile_path
            else default_user_profile_path(user_name)
        )
        profile_paths = {profile_name: target_profile}
    else:
        if profile_name.casefold() != "default":
            parser.error("The legacy You profile has only the default profile; configure named profiles in config.yaml")
        profile_name = "default"
        target_profile = (
            Path(config.speaker_verification.profile_path).expanduser()
            if config.speaker_verification.profile_path
            else default_profile_path()
        )
        profile_paths = {profile_name: target_profile}
    verifier = SpeakerVerifier(
        profile_path=target_profile,
        threshold=config.speaker_verification.threshold,
        user_profiles={user_name: profile_paths},
    )
    append = False
    if verifier.enrolled:
        if args.replace:
            answer = input(f"Replace {user_name}'s existing speaker profile at {target_profile}? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Enrollment cancelled; existing profile was left unchanged.")
                return
        else:
            answer = input(
                f"Add more '{profile_name}' samples to {user_name}'s profile at {target_profile}? [Y/n] "
            ).strip().lower()
            if answer in {"n", "no"}:
                print("Enrollment cancelled; existing profile was left unchanged.")
                return
            append = True

    print(f"Enrollment records three short samples for {user_name}'s '{profile_name}' profile.")
    guidance = PROFILE_GUIDANCE.get(
        profile_name.casefold(),
        f"Position the mic and yourself to match the '{profile_name}' condition.",
    )
    print(f"{guidance} Keep the room representative of actual use.")
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
                input(
                    f"\nSample {index}/{len(PROMPTS)} ({profile_name}). Press Enter when ready, then say: {prompt} "
                )
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
        verifier.enroll(
            samples,
            append=append,
            user_name=user_name,
            profile_name=profile_name,
        )
    except Exception as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        raise
    finally:
        samples.clear()
        gc.collect()

    print(
        f"Speaker profile saved locally at {target_profile} "
        f"({len(verifier.profiles[user_name][profile_name])} examples for {user_name}/{profile_name})."
    )
    print("Restart Adam to enable speaker checks. To remove the profile, delete that file.")


if __name__ == "__main__":
    main()
