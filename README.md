# Adam Voice Assistant

Adam is an asynchronous voice assistant for Linux desktop sessions. It can use
local or hosted speech and language models, speak replies, and optionally control
desktop apps or record meetings.

## Quick start

You need a Linux user session, a working microphone and audio output, and
[`uv`](https://docs.astral.sh/uv/). Setup uses uv to provide Python 3.13 or newer
and create the project environment. Run setup as your normal desktop user; do not
run it as root.

Choose a provider before setup. If you want a local LLM, install and start Ollama
yourself; Adam's setup does not install Ollama. Setup can offer to download the
model you choose. For a hosted speech or LLM provider, have its API key ready.

From the repository directory:

```sh
./setup.sh --dry-run
./setup.sh
```

The dry run makes no changes. Interactive setup lets you choose audio devices,
wake phrase, speech recognition, speech output, LLM provider, optional voice
enrollment and meeting storage, and whether to create a systemd user service.
If `config.yaml` does not exist, setup copies `config.yaml.example` and sets
owner-only permissions (`0600`). It does not replace an existing config; wizard
choices update the settings you select.

CPU is the default compute path. GPU acceleration is optional; choose the
specific supported device in setup. Model downloads are optional. Use
`--skip-models` to skip the default speech and voice asset downloads.

Useful setup options:

```sh
./setup.sh --help
./setup.sh --yes                 # noninteractive bootstrap; does not start Adam
./setup.sh --cpu-only
./setup.sh --skip-models
```

### Start Adam

If setup created the systemd user unit, start it and check its status with:

```sh
systemctl --user enable --now adam.service
systemctl --user status adam.service
journalctl --user -u adam.service -f
```

Or run Adam in the foreground from the repository directory:

```sh
uv run python -m src.main
```

Say **“Hey Adam”** followed by a request. If you chose another wake phrase,
use that phrase instead. On systems without systemd, run Adam in the foreground
or configure a service for your init system.

## Choose speech and language providers

The setup wizard can configure local or hosted speech recognition and speech
output, plus a local or hosted language model. Local model services are separate
applications: for example, Adam connects to an Ollama server that you install
and run yourself. Hosted providers require their API keys.

Qwen3-ASR detects the spoken language automatically; it does not accept a manual
English language override. Faster-Whisper `small.en` is available on CPU or a
selected NVIDIA GPU. Hosted transcription can use OpenAI, OpenRouter, or a
custom OpenAI-compatible endpoint.

Provider credentials entered in setup stay in your local `config.yaml`, which
is created with owner-only permissions. Choosing a cloud provider sends the
audio or text needed for that request to that provider. Keep `config.yaml`
private and do not commit it.

## Meetings and voice profiles

Start and stop recording with the exact voice commands **“Hey Adam, meeting
mode on”** and **“Hey Adam, meeting mode off.”** The assistant can continue to
handle ordinary requests while recording. A meeting stops at its configured
duration limit (four hours by default).

Meeting audio remains the original mixed recording. Optional Nemotron
diarization can split it into speaker turns; it does not isolate or remove
participants. Speaker names and anonymous labels are best-effort: without
diarization, Adam labels whole speech segments using voice similarity, and a
speaker may be marked `Unknown`.

By default, meeting files are saved under
`~/.local/state/adam/meetings/`. Adam also saves each finalized microphone
capture and transcript stages under `~/.local/state/adam/heard-captures/`.
These files can contain private conversations. Recording directories and files
are created with owner-only permissions; remove them when you no longer need
them.

When voice verification is enabled, enroll a profile with:

```sh
uv run python -m src.stt.enroll
```

Named users and additional profiles can be configured under
`speaker_verification.users` in `config.yaml`; see the example configuration.
Restart Adam after enrollment to load the updated profile.

## Desktop and browser use

Desktop control begins with a fresh observation of the focused window. Adam can
perform bounded actions or a short sequence, checking the screen again between
steps and asking for a new target decision when the view changes. Input and
observation support depends on the active X11 or Wayland session and installed
desktop tools. Disable it with `computer_control.enabled: false` in
`config.yaml`.

Optional browser navigation uses a separate Adam browser profile. It can inspect
visible page text, open URLs or searches, follow links, fill ordinary text
fields, scroll, and use history. It does not press buttons or submit forms.
Enable it with `./setup.sh --browser-navigation` or choose it in interactive
setup. Treat web page content as untrusted instructions.

Optional OmniParser screenshot grounding adds numbered candidate control boxes;
it does not identify their text. Setup can install its isolated CPU or NVIDIA
runtime. The YOLOv8 checkpoint and Ultralytics runtime are AGPL-3.0 licensed.

## Web UI

The optional Web UI provides text interaction, live tool activity, and system
status in a local browser. It is disabled by default and listens on loopback.
It does not bypass voice confirmation for sensitive actions.

For one foreground session, start Adam with the UI enabled:

```sh
uv run python -m src.main --webui
```

Then open <http://127.0.0.1:8765>. To enable it for a systemd service, set
`webui.enabled: true` in `config.yaml`, then run
`systemctl --user restart adam.service`. See
[the Web UI guide](docs/webui-sidecar-setup.md) for configuration and security
details.

## Wake-free idea routing

Idea routing is an optional local feature that can recognize selected direct
requests without a wake phrase. It uses local speech recognition and an enrolled
voice profile to reduce responses to other speakers. Enable it with
`./setup.sh --idea-routing`; edit the ideas in `assets/intent_ideas.json` and
adjust thresholds in `config.yaml`.

## Configuration and development

Use `config.yaml.example` as the reference for available settings. The local
`config.yaml` contains host-specific choices and credentials. Setup preserves
it unless you change a setting through the wizard.

For implementation notes, evaluations, and the Web UI details, see
[`docs/`](docs/). To run the test suite from a prepared environment:

```sh
uv run pytest
```
