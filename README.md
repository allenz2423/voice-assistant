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
./setup.sh --idea-routing
./setup.sh --browser-navigation
```

The setup script can prepare a systemd user service where systemd is available.
On other Linux init systems, launch Adam in the foreground with
`uv run python -m src.main` or create a service using your init system.

## Wake-free idea routing

During idle listening, Adam can locally transcribe speech and compare it with
the ideas in `assets/intent_ideas.json`. This can route direct computer requests
such as “Open Spotify” to Adam's regular assistant flow without a wake phrase.
Future personal calendar statements can trigger a separate LLM review that has
access only to the calendar creation tool. That review requires a definite
future event and date/time; it does not summarize or respond conversationally.

Enable the feature with `./setup.sh --idea-routing` or answer yes to the optional
setup question. Setup installs the small `all-MiniLM-L6-v2` quantized ONNX model
and asks for the speaker profile needed to ignore other voices. The model runs
on CPU, even when Adam's speech recognition uses a GPU. Idle audio for this
feature is transcribed with the local ASR path, and idea routing pauses while
meeting mode is active. You can turn it off with `idea_routing.enabled: false`
in `config.yaml`.

Ideas are concepts, not a list of command examples. Edit their titles,
descriptions, embedding text, and route in `assets/intent_ideas.json`; use the
thresholds in `config.yaml` to adjust how readily each route is considered.
Background calendar candidates use a stricter threshold and still require the
isolated LLM review. The checked-in 2,000-phrase corpus is synthetic and is a
repeatable calibration check, not a measure of real-world accuracy:

To save a personal phrase for future semantic matching, say `Make a memory,
<your exact phrase>` or `Make a memory that <your exact phrase>`. Adam stores the recognized text after the
command phrase in `~/.local/state/adam/embedding-memories.json` without
rewriting it. A later
similar idle utterance must clear a separate high similarity threshold before
Adam wakes the normal LLM, with the matched memory supplied as context; the
memory itself does not authorize an action.

```sh
uv run python tools/evaluate_idea_router.py
uv run pytest tests/test_idea_router.py tests/test_background_idea_review.py
```

The ONNX model is downloaded to the user cache rather than stored in the repo.
If setup skipped the download, run `uv run python -m src.intent.idea_router
--download` after installing the optional dependencies.

## Browser navigation

Enable browser navigation with `./setup.sh --browser-navigation` or accept the
optional wizard prompt. Adam can inspect visible page text, open HTTP(S) URLs or
searches, follow listed links, fill ordinary text fields, scroll, and use browser
history. It does not press buttons or submit forms. Treat page content as
untrusted; Adam's instructions tell it to follow your request rather than text
embedded in a page.

The browser opens visibly in a dedicated profile at
`~/.local/share/adam/browser-navigation`, separate from your regular browser
profile. Sign in to websites in that Adam profile if needed. The wizard supports
your configured Chromium browser (such as Microsoft Edge or Google Chrome),
Playwright's bundled Chromium, or Firefox. Playwright can launch branded Chrome
and Edge using its documented `chrome` and `msedge` channels, and its managed
Firefox build; the wizard installs the managed browser for Firefox and bundled
Chromium choices ([Playwright browser launch documentation](https://playwright.dev/python/docs/api/class-browsertype)).

You can change `browser_navigation.enabled` and `browser_navigation.browser` in
`config.yaml`. Valid browser values are `default`, `chromium`, or `firefox`.
To run the browser checks locally, use `uv run pytest tests/test_browser_navigation.py`;
the test page is served from localhost and exercises navigation, safe field entry,
history, and URL filtering in Chromium and Firefox (plus Edge when installed).

## Visual desktop control

Adam can inspect the current desktop screenshot, then perform one bounded click,
text entry, key press, or scroll at a time. Each input must cite the ID from the
latest screenshot, and every input returns a new screenshot for the next choice.
The controller detects Wayland or X11 from the live session: on Wayland it uses
`ydotool` for mouse input and `wtype` for keyboard input; on X11 it uses
`xdotool`. Wayland mouse control starts the user `ydotool.service` on demand.

Visual control is enabled by default when matching input tools are installed;
set `computer_control.enabled: false` in `config.yaml` to disable it. The setup
script installs the desktop input tools on supported distros. Routing and stale
screenshot checks can be tested without a desktop using
`uv run pytest tests/test_computer_control.py`. Adam's instructions require a
fresh screenshot between actions, confirmation before purchases and external
submissions, and prohibit sharing intimate or private content.

Optional screenshot grounding uses OmniParser's YOLOv8 Nano interactable-region
detector. It draws numbered boxes and reports screenshot-pixel centers; it does
not provide text labels, so Adam still reads the attached screenshot or
accessibility data to choose a box. Its roughly 41 MB weights run in an isolated
PyTorch 2.6 and Ultralytics runtime. Setup lets you choose CPU or one exact
NVIDIA GPU UUID; Vulkan is not supported. The YOLOv8 checkpoint and Ultralytics
runtime are AGPL-3.0 licensed.

## Meeting mode

For deterministic voice controls, say **“Hey Adam, meeting mode on”** to start
continuous capture and **“Hey Adam, meeting mode off”** to stop. Natural requests
such as “start meeting mode” or “meeting over” go through Brain and its
`meeting_mode` tool. Adam also continues to listen for ordinary commands during
the meeting. Sessions stop at the configured time limit (four hours by default).

Meeting audio is stored as the original mixed recording. Diarization adds speaker
labels to transcript turns; it does not isolate or suppress other people. When
speaker verification is enabled, configured users receive their configured
names in meeting transcripts. Other participants receive stable anonymous
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

## Speaker enrollment

Adam supports separate named users with distinct acoustic profiles. Configure
them under `speaker_verification.users` in `config.yaml`; default files go under
`~/.local/state/adam/speaker-profiles/`. Profile names can be a short list or a
map to custom paths. For example:

```yaml
speaker_verification:
  enabled: true
  threshold: 0.25
  users:
    - name: Allen
      profiles: [near, close, far]
    - name: Alex
      profiles:
        near: "~/.local/state/adam/speaker-profiles/alex/near.npz"
        far: "~/.local/state/adam/speaker-profiles/alex/far.npz"
```

Enroll a selected user/profile with, for example,
`uv run python -m src.stt.enroll --user Allen --profile far`. Each run records
three samples only for that acoustic condition. Re-running adds examples to that
user's selected profile; pass `--replace` to replace just that profile. Profiles
for one person remain grouped under one configured user, so meeting transcripts
use **Allen** regardless of which distance profile matched. When `users` is
empty, the existing single-user `speaker-profile.npz` remains supported as
**You**; once users are listed, only those configured profiles are used. Only
voice embeddings are saved, not enrollment recordings. Restart Adam after
enrollment to load updated profiles.

## Web dashboard (Sidecar)

Adam includes an opt-in, loopback-only browser interface for text interaction,
real-time tool activity inspection, and system status monitoring alongside
normal voice interaction. The Web UI is disabled by default and preserves
microphone-first verbal confirmation for sensitive actions.

Enable it with `python -m src.main --webui` or set `webui.enabled: true` in
`config.yaml`. See [docs/webui-sidecar-setup.md](docs/webui-sidecar-setup.md) for
the interface layout, keyboard controls, security policies, and setup details.

## Development

Run checks with:

```sh
uv run pytest
```

The optional evaluation scripts under `tools/` compare diarization and meeting
transcripts against AMI corpus references. Downloaded recordings and local
speaker-profile experiments are not part of the source repository.
