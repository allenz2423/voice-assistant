"""Continuous meeting audio capture with VAD-delimited transcription work."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


class MeetingSession:
    """Write the full mic stream while asynchronously queuing speech turns."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int,
        max_duration_seconds: float = 4 * 60 * 60,
        silence_duration: float = 1.25,
        speech_threshold: float = 0.25,
        max_segment_seconds: float = 30.0,
        process_segment: Callable[[np.ndarray, np.ndarray, float, float], None] | None = None,
        vad_factory: Callable[..., object] | None = None,
        audio_queue_size: int = 256,
        segment_queue_size: int = 64,
    ):
        self.root = Path(root).expanduser()
        self.sample_rate = int(sample_rate)
        self.max_duration_seconds = max(1.0, float(max_duration_seconds))
        self.silence_duration = max(0.2, float(silence_duration))
        self.speech_threshold = float(speech_threshold)
        self.max_segment_seconds = max(2.0, float(max_segment_seconds))
        self.process_segment = process_segment
        self.vad_factory = vad_factory
        self.audio_queue_size = max(8, int(audio_queue_size))
        self.segment_queue_size = max(1, int(segment_queue_size))

        self.session_dir: Path | None = None
        self.audio_path: Path | None = None
        self.transcript_path: Path | None = None
        self.turns_path: Path | None = None
        self.started_at: datetime | None = None
        self._started_monotonic = 0.0
        self._accepting = False
        self._lock = threading.RLock()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=self.audio_queue_size)
        self._segment_queue: queue.Queue = queue.Queue(maxsize=self.segment_queue_size)
        self._audio_thread: threading.Thread | None = None
        self._segment_thread: threading.Thread | None = None
        self._audio_drops = 0
        self._transcript_drops = 0
        self._written_samples = 0
        self._stop_reason = ""

    @property
    def active(self) -> bool:
        with self._lock:
            return self._accepting

    @property
    def elapsed_seconds(self) -> float:
        if not self._started_monotonic:
            return 0.0
        return max(0.0, time.monotonic() - self._started_monotonic)

    @property
    def max_duration_reached(self) -> bool:
        return self.active and self.elapsed_seconds >= self.max_duration_seconds

    def start(self) -> Path:
        with self._lock:
            if self._accepting:
                raise RuntimeError("A meeting session is already active")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            self.started_at = datetime.now().astimezone()
            stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
            self.session_dir = self.root / f"meeting-{stamp}"
            suffix = 1
            while self.session_dir.exists():
                self.session_dir = self.root / f"meeting-{stamp}-{suffix}"
                suffix += 1
            self.session_dir.mkdir(mode=0o700)
            self.audio_path = self.session_dir / "meeting.wav"
            self.transcript_path = self.session_dir / "transcript.txt"
            self.turns_path = self.session_dir / "turns.jsonl"
            self.transcript_path.touch(mode=0o600)
            self.turns_path.touch(mode=0o600)
            os.chmod(self.transcript_path, 0o600)
            os.chmod(self.turns_path, 0o600)
            self._started_monotonic = time.monotonic()
            self._audio_drops = self._transcript_drops = self._written_samples = 0
            self._stop_reason = ""
            self._audio_queue = queue.Queue(maxsize=self.audio_queue_size)
            self._segment_queue = queue.Queue(maxsize=self.segment_queue_size)
            self._accepting = True
            self._audio_thread = threading.Thread(
                target=self._write_and_segment_audio, name="adam-meeting-audio", daemon=True
            )
            self._segment_thread = threading.Thread(
                target=self._process_segments, name="adam-meeting-transcription", daemon=True
            )
            self._audio_thread.start()
            self._segment_thread.start()
            metadata = {
                "started_at": self.started_at.isoformat(),
                "sample_rate": self.sample_rate,
                "max_duration_seconds": self.max_duration_seconds,
                "audio_file": self.audio_path.name,
                "transcript_file": self.transcript_path.name,
                "speaker_turns_file": self.turns_path.name,
            }
            self._write_metadata(metadata)
            return self.session_dir

    def enqueue_audio(self, audio: np.ndarray) -> bool:
        """Nonblocking audio callback for AudioStreamManager's capture thread."""
        with self._lock:
            if not self._accepting:
                return False
        if self.elapsed_seconds >= self.max_duration_seconds:
            return False
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return True
        try:
            self._audio_queue.put_nowait(chunk.copy())
            return True
        except queue.Full:
            with self._lock:
                self._audio_drops += 1
            return False

    def append_turn(
        self,
        speaker: str,
        text: str,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        text = " ".join(str(text or "").split())
        if not text or self.session_dir is None:
            return
        elapsed = self.elapsed_seconds if start_seconds is None else max(0.0, float(start_seconds))
        end = elapsed if end_seconds is None else max(elapsed, float(end_seconds))
        turn = {
            "start_seconds": round(elapsed, 3),
            "end_seconds": round(end, 3),
            "speaker": str(speaker or "Unknown"),
            "text": text,
        }
        stamp = self._format_timestamp(elapsed)
        with self._lock:
            if self.turns_path is None or self.transcript_path is None:
                return
            with self.turns_path.open("a", encoding="utf-8") as turns_file:
                turns_file.write(json.dumps(turn, ensure_ascii=False) + "\n")
            with self.transcript_path.open("a", encoding="utf-8") as transcript_file:
                transcript_file.write(f"[{stamp}] {turn['speaker']}: {text}\n")
            os.chmod(self.turns_path, 0o600)
            os.chmod(self.transcript_path, 0o600)

    def stop(self, reason: str = "user") -> Path | None:
        with self._lock:
            if not self._accepting:
                return self.session_dir
            self._accepting = False
            self._stop_reason = reason
        # The audio thread drains all queued frames, finalizes its last speech
        # segment, and then closes the WAV before returning.
        if self._audio_thread:
            # The audio consumer normally drains a full queue. If it has already
            # failed, avoid waiting forever to enqueue a sentinel nobody can read.
            while self._audio_thread.is_alive():
                try:
                    self._audio_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._audio_thread.join()
        if self._segment_thread:
            self._segment_thread.join()
        with self._lock:
            metadata = self._read_metadata()
            metadata.update({
                "ended_at": datetime.now().astimezone().isoformat(),
                "duration_seconds": round(self.elapsed_seconds, 3),
                "stop_reason": reason,
                "audio_queue_dropped_chunks": self._audio_drops,
                "transcription_queue_dropped_segments": self._transcript_drops,
                "written_audio_seconds": round(self._written_samples / self.sample_rate, 3),
            })
            self._write_metadata(metadata)
            return self.session_dir

    def _write_and_segment_audio(self) -> None:
        try:
            self._write_and_segment_audio_inner()
        except Exception as exc:
            print(f"[Meeting] Audio segmentation failed; recording may be incomplete ({exc}).", flush=True)
        finally:
            # Always release the transcription worker, including when VAD or WAV
            # setup fails, so stopping a meeting cannot hang on a dead worker.
            self._segment_queue.put(None)

    def _write_and_segment_audio_inner(self) -> None:
        from src.audio.vad import SileroVAD

        vad = self.vad_factory(sample_rate=self.sample_rate) if self.vad_factory else SileroVAD(sample_rate=self.sample_rate)
        pre_roll: deque[np.ndarray] = deque()
        pre_roll_samples = 0
        context: deque[np.ndarray] = deque()
        context_samples = 0
        context_limit = int(5.0 * self.sample_rate)
        pre_roll_limit = int(0.32 * self.sample_rate)
        frames: list[np.ndarray] = []
        segment_context = np.zeros(0, dtype=np.float32)
        segment_start = 0.0
        segment_samples = 0
        silence_samples = 0
        last_speech_index = 0
        speech_active = False
        max_segment_samples = int(self.max_segment_seconds * self.sample_rate)

        def finalize_segment() -> None:
            nonlocal frames, segment_context, segment_start, segment_samples
            nonlocal silence_samples, last_speech_index, speech_active
            if last_speech_index > 0:
                audio = np.concatenate(frames[:last_speech_index]).astype(np.float32, copy=False)
                end = segment_start + len(audio) / self.sample_rate
                try:
                    self._segment_queue.put_nowait((audio, segment_context, segment_start, end))
                except queue.Full:
                    with self._lock:
                        self._transcript_drops += 1
                    print("[Meeting] Transcription queue full; this speech segment remains in the WAV only.", flush=True)
            frames = []
            segment_context = np.zeros(0, dtype=np.float32)
            segment_samples = silence_samples = last_speech_index = 0
            speech_active = False

        assert self.audio_path is not None
        with wave.open(str(self.audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            os.chmod(self.audio_path, 0o600)
            while True:
                chunk = self._audio_queue.get()
                if chunk is None:
                    break
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2")
                wav_file.writeframesraw(pcm.tobytes())
                self._written_samples += len(chunk)

                is_speech, _ = vad.is_speech(
                    chunk, threshold=self.speech_threshold, use_energy_floor=True, energy_floor=0.0025
                )
                if is_speech:
                    if not speech_active:
                        speech_active = True
                        segment_context = np.concatenate(list(context)) if context else np.zeros(0, dtype=np.float32)
                        segment_start = max(0.0, (self._written_samples - len(chunk) - pre_roll_samples) / self.sample_rate)
                        frames = list(pre_roll)
                        segment_samples = sum(map(len, frames))
                        last_speech_index = len(frames)
                    frames.append(chunk)
                    segment_samples += len(chunk)
                    last_speech_index = len(frames)
                    silence_samples = 0
                    if segment_samples >= max_segment_samples:
                        finalize_segment()
                elif speech_active:
                    frames.append(chunk)
                    segment_samples += len(chunk)
                    silence_samples += len(chunk)
                    if silence_samples >= int(self.silence_duration * self.sample_rate):
                        finalize_segment()

                context.append(chunk)
                context_samples += len(chunk)
                while context_samples > context_limit and context:
                    removed = context.popleft()
                    context_samples -= len(removed)
                if not speech_active:
                    pre_roll.append(chunk)
                    pre_roll_samples += len(chunk)
                    while pre_roll_samples > pre_roll_limit and pre_roll:
                        removed = pre_roll.popleft()
                        pre_roll_samples -= len(removed)
            if speech_active:
                finalize_segment()

    def _process_segments(self) -> None:
        while True:
            item = self._segment_queue.get()
            if item is None:
                return
            if self.process_segment is None:
                continue
            try:
                self.process_segment(*item)
            except Exception as exc:
                print(f"[Meeting] Could not transcribe/label a speech segment ({exc}).", flush=True)

    def _metadata_path(self) -> Path:
        assert self.session_dir is not None
        return self.session_dir / "session.json"

    def _write_metadata(self, metadata: dict) -> None:
        if self.session_dir is None:
            return
        path = self._metadata_path()
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)

    def _read_metadata(self) -> dict:
        try:
            return json.loads(self._metadata_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
