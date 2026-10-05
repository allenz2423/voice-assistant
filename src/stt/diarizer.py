"""Local NVIDIA Nemotron-3 speaker diarization through Transformers."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpeakerSpan:
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerTurn:
    speakers: tuple[str, ...]
    start: float
    end: float
    audio: np.ndarray


class NemotronDiarizer:
    """Diarize mono 16 kHz audio with NVIDIA's official Transformers model."""

    def __init__(self, executable: str = "nemo-speech", model: str = "nvidia/Nemotron-3-Diarization",
                 device: str = "auto", timeout: float = 45.0):
        # `executable` is retained for config compatibility with older installs.
        self.executable = executable
        self.model_id = model
        self.device = device.lower().strip()
        self.timeout = timeout
        self._processor = None
        self._model = None
        self._torch = None

    @property
    def available(self) -> bool:
        # Availability is checked during daemon startup. Inspect package specs
        # here so the optional Transformers/PyTorch stack stays unloaded until
        # diarization is actually requested.
        try:
            return all(
                package in sys.modules or importlib.util.find_spec(package) is not None
                for package in ("transformers", "torch")
            )
        except (ImportError, ValueError):
            return False

    def supports_model(self) -> bool:
        """Transformers accepts either a Hub model ID or a local model folder."""
        if not self.model_id or not self.model_id.strip():
            return False
        model_path = Path(self.model_id).expanduser()
        if model_path.is_dir():
            return (model_path / "config.json").is_file() and (model_path / "model.safetensors").is_file()
        if model_path.is_file():
            return False
        try:
            from huggingface_hub import try_to_load_from_cache
            return all(
                try_to_load_from_cache(self.model_id, name) is not None
                for name in ("config.json", "processor_config.json", "model.safetensors")
            )
        except (ImportError, OSError, ValueError):
            return False

    @staticmethod
    def _cuda_supports_device(torch, device: str) -> tuple[bool, str | None]:
        """Check whether this PyTorch build has kernels for the selected GPU."""
        try:
            index = int(device.split(":", 1)[1]) if ":" in device else int(torch.cuda.current_device())
            capability = torch.cuda.get_device_capability(index)
            architecture = f"sm_{capability[0]}{capability[1]}"
            compiled_architectures = torch.cuda.get_arch_list()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            return False, str(exc)
        if architecture not in compiled_architectures:
            return False, f"PyTorch has no kernels for {architecture}"
        return True, None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Nemotron requires the optional Transformers diarization runtime; "
                "rerun setup and install it."
            ) from exc

        requested = self.device
        # Never guess which GPU the user intended. In particular, old Vulkan
        # settings came from a user-selected device (often not PyTorch's CUDA 0).
        # This Transformers runtime cannot use Vulkan, so keep those configs on CPU.
        if requested.startswith("vulkan"):
            print("[Diarization] Vulkan is unavailable in the Transformers runtime; using CPU.", flush=True)
            selected = "cpu"
        elif requested in ("", "auto"):
            selected = "cpu"
        elif requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was selected, but PyTorch cannot access an NVIDIA GPU.")
            selected = requested
        elif requested == "cpu":
            selected = "cpu"
        else:
            raise ValueError(f"Unsupported Nemotron device {requested!r}; choose auto, cpu, or cuda[:index].")

        if selected.startswith("cuda"):
            supported, reason = self._cuda_supports_device(torch, selected)
            if not supported:
                if requested in ("", "auto"):
                    print(f"[Diarization] {reason}; using CPU.", flush=True)
                    selected = "cpu"
                else:
                    raise RuntimeError(f"Nemotron CUDA device {selected} is unsupported: {reason}.")

        dtype = torch.float16 if selected.startswith("cuda") else torch.float32
        # Setup downloads the complete snapshot in advance. Offline loading keeps
        # daemon startup/inference from blocking on Hub retry loops.
        self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
        self._model = AutoModelForAudioFrameClassification.from_pretrained(
            self.model_id, dtype=dtype, local_files_only=True
        ).to(selected)
        self._model.eval()
        self._torch = torch

    def diarize(self, audio: np.ndarray) -> list[SpeakerSpan]:
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if source.size == 0:
            return []
        self._load()
        sample_rate = int(self._processor.feature_extractor.sampling_rate)
        if sample_rate != 16000:
            raise ValueError(f"Nemotron expects 16 kHz audio; processor requests {sample_rate} Hz")
        inputs = self._processor(source, sampling_rate=sample_rate)
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        segments = self._processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        return [
            SpeakerSpan(f"speaker_{int(segment['Speaker'])}", float(segment["Start"]), float(segment["End"]))
            for segment in segments
            if float(segment["End"]) > float(segment["Start"])
        ]

    @staticmethod
    def exclusive_speaker_audio(audio: np.ndarray, spans: list[SpeakerSpan], sample_rate: int,
                                min_seconds: float = 0.5) -> dict[str, np.ndarray]:
        """Return audio where each speaker is active alone; overlapping frames are omitted."""
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("Nemotron diarization expects 16 kHz audio")
        frame_count = len(source)
        activity: dict[str, np.ndarray] = {}
        for span in spans:
            start = max(0, min(frame_count, int(span.start * sample_rate)))
            end = max(start, min(frame_count, int(span.end * sample_rate)))
            if end > start:
                activity.setdefault(span.speaker, np.zeros(frame_count, dtype=bool))[start:end] = True
        if not activity:
            return {}
        active_count = np.sum(np.stack(list(activity.values())), axis=0)
        result: dict[str, np.ndarray] = {}
        minimum = int(min_seconds * sample_rate)
        for speaker, mask in activity.items():
            selected = source[mask & (active_count == 1)]
            if len(selected) >= minimum:
                result[speaker] = selected
        return result

    @staticmethod
    def speaker_turns(audio: np.ndarray, spans: list[SpeakerSpan], sample_rate: int,
                      min_seconds: float = 0.1, merge_gap: float = 3.0,
                      short_overlap_seconds: float = 0.4) -> list[SpeakerTurn]:
        """Split the timeline by diarized activity, preserving every audio sample.

        Unlike source separation, each turn contains an untouched crop of the
        original mix. Simultaneous speakers are represented together in
        ``speakers`` so their audio and transcript are not silently discarded.
        """
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("Nemotron diarization expects 16 kHz audio")
        if not source.size:
            return []

        duration = len(source) / sample_rate
        clipped = [
            (span.speaker, max(0.0, min(duration, float(span.start))),
             max(0.0, min(duration, float(span.end))))
            for span in spans
            if span.end > span.start
        ]
        clipped = [span for span in clipped if span[2] > span[1]]
        boundaries = sorted({point for _, start, end in clipped for point in (start, end)})
        intervals: list[tuple[tuple[str, ...], float, float]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            midpoint = (start + end) / 2.0
            active = tuple(sorted({
                speaker for speaker, span_start, span_end in clipped
                if span_start <= midpoint < span_end
            }))
            if not active:
                continue
            intervals.append((active, start, end))

        # A tiny overlap can split one otherwise continuous ASR turn into
        # sub-second fragments. Attribute that brief interval to the speaker
        # who was active on both sides; longer overlaps stay explicitly shared.
        for index in range(1, len(intervals) - 1):
            speakers, start, end = intervals[index]
            previous = intervals[index - 1][0]
            following = intervals[index + 1][0]
            if (
                len(speakers) > 1
                and end - start <= short_overlap_seconds
                and len(previous) == len(following) == 1
                and previous == following
                and previous[0] in speakers
            ):
                intervals[index] = (previous, start, end)

        merged_intervals: list[tuple[tuple[str, ...], float, float]] = []
        for speakers, start, end in intervals:
            if (
                merged_intervals
                and merged_intervals[-1][0] == speakers
                and start - merged_intervals[-1][2] <= merge_gap
            ):
                previous_speakers, previous_start, _ = merged_intervals[-1]
                merged_intervals[-1] = (previous_speakers, previous_start, end)
            else:
                merged_intervals.append((speakers, start, end))

        minimum = max(0, int(min_seconds * sample_rate))
        turns = []
        for speakers, start, end in merged_intervals:
            start_sample = max(0, min(len(source), int(start * sample_rate)))
            end_sample = max(start_sample, min(len(source), int(end * sample_rate)))
            if end_sample - start_sample >= minimum:
                turns.append(SpeakerTurn(speakers, start, end, source[start_sample:end_sample].copy()))
        return turns
