# voice-assistant
# Adam Voice Assistant

Adam is a Linux voice assistant with local or cloud speech recognition, a
speaker profile, desktop controls, and an optional continuous meeting mode.

## Install

From the repository directory, run:

```sh
./setup.sh
```

The interactive setup detects the Linux package manager and offers choices for
audio devices, speech recognition, compute devices, text-to-speech, LLM provider,
speaker verification, meeting storage, optional diarization, and the user service.
It supports Debian and Ubuntu families (`apt`), Fedora and RHEL families (`dnf`),
openSUSE (`zypper`), Arch families (`pacman`), Gentoo (`emerge`), Alpine (`apk`),
Void (`xbps`), and Solus (`eopkg`). NixOS is detected and given package guidance
without modifying its declarative system configuration. Unsupported package names
are retried individually so an optional desktop tool does not stop the whole setup.
Alpine's musl-based Python environment may not have wheels for every ML dependency;
use a glibc-based distro or container if `uv sync` reports an unsupported wheel.

The speech menu includes:

- Qwen3-ASR on a specifically selected Vulkan GPU, with English forcing or
  multilingual language detection.
- Faster-Whisper `small.en` on an explicitly selected NVIDIA GPU.
- Faster-Whisper `small.en` on CPU for laptops without a supported GPU.
- Cloud and custom OpenAI-compatible transcription providers.

GPU runtime libraries are installed only when requested or when setup detects an
NVIDIA GPU and the user accepts. Adam never chooses between multiple GPUs without
showing the device names and asking for a selection. For an explicitly CPU-only
install, run `./setup.sh --cpu-only`. Use `./setup.sh --nvidia-runtime` to install
NVIDIA runtime support directly. Existing `config.yaml` settings are preserved.

Useful setup options:

```sh
./setup.sh --help
./setup.sh --cpu-only
./setup.sh --skip-sys-pkgs --skip-models --skip-service
./setup.sh --skip-speaker-verification --skip-enrollment
```

The setup script can prepare a systemd user service where systemd is available.
On other Linux init systems, launch Adam in the foreground with
`uv run python -m src.main` or create a service using your init system.

## Meeting mode

Say **“Hey Adam, meeting mode”** to start continuous capture and transcription.
Say **“Hey Adam, meeting over”** to stop. Adam also continues to listen for
ordinary commands during the meeting. Sessions stop at the configured time limit
(four hours by default).

Meeting audio is stored as the original mixed recording. Diarization adds speaker
labels to transcript turns; it does not isolate or suppress other people. When
speaker verification is enabled and a voice profile exists, Adam labels the
enrolled participant as **You**. Other participants receive stable anonymous
labels. Nemotron diarization is optional and can be run on CPU or an explicitly
selected NVIDIA GPU.

Meeting recordings and transcripts default to
`~/.local/state/adam/meetings/`. Adam also records finalized microphone captures
and transcript stages in `~/.local/state/adam/heard-captures/`. These paths can
contain private conversations; keep them local and remove them when no longer
needed. Recording directories are created with owner-only permissions.

Each meeting directory includes `meeting.wav` (original audio), `transcript.txt`
(readable speaker-labeled text), `turns.jsonl` (speaker labels and timestamps),
and `session.json` (session metadata).

## Development

Run checks with:

```sh
uv run pytest
```

The optional evaluation scripts under `tools/` compare diarization and meeting
transcripts against AMI corpus references. Downloaded recordings and local
speaker-profile experiments are not part of the source repository.
