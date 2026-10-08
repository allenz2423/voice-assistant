# Adam Voice Assistant

Adam is an asynchronous voice assistant for Linux desktop sessions. It supports
local and hosted speech recognition, language models, and speech output. Desktop
control, meeting recording, browser navigation, and the Web UI are optional and
can be configured for each host.

## Install

Run setup as the regular desktop user who will run Adam. Do not run it as root.
Install [Astral uv](https://docs.astral.sh/uv/getting-started/installation/)
first; uv manages the Python 3.13+ environment. From the repository directory:

```sh
./setup.sh --dry-run
./setup.sh
```

The dry run prints the planned setup without running commands or changing files.
Interactive setup installs required system and Python dependencies, offers
optional desktop-control packages, then guides you through audio devices,
speech and language providers, voice enrollment, meeting settings, optional
features, and systemd service setup. It may download model assets after asking.
Setup creates `config.yaml` from `config.yaml.example` when needed, protects it
with owner-only permissions, and preserves an existing configuration except
for settings explicitly selected in setup.

The default Python runtime is CPU-based. NVIDIA runtime packages are optional;
use `--nvidia-runtime` to select them or `--cpu-only` to force CPU choices in
the wizard. Desktop-control packages and optional features are not required for
voice use. See `./setup.sh --help` for all options, including `--skip-sys-pkgs`,
`--skip-python-deps`, `--skip-models`, and `--skip-service`.

`./setup.sh --yes` performs a noninteractive bootstrap. It applies requested
feature flags, uses the CPU runtime unless `--nvidia-runtime` is selected, and
downloads the default Kokoro speech assets unless `--skip-models` is given. It
does not run the configuration wizard or enable/start a service. If systemd
user setup is available, it installs a unit only when one does not already
exist. By contrast, interactive setup asks whether to install and enable/start
the service. If you skip Kokoro assets while `tts.engine` is still `kokoro`,
download those files or choose another speech engine before starting Adam.

## Choose providers

The example configuration uses local Faster-Whisper `small.en` speech
recognition on CPU, Kokoro CPU speech output, and a local Ollama language model.
The setup wizard can select hosted speech/LLM providers or other supported local
engines. Adam does not install the Ollama server; install and run it separately
if you choose Ollama. Interactive setup may offer to pull the selected model
when Ollama is already available. Hosted providers require their API keys.

Provider credentials are stored in the private local `config.yaml`. Requests to
a hosted speech or language provider send the audio or text needed for that
request to the provider. Keep `config.yaml` private and do not commit it.

## Start Adam

To run in the foreground from the repository directory:

```sh
uv run python -m src.main
```

If interactive setup installed and enabled the systemd user service, manage it
with:

```sh
systemctl --user status adam.service
journalctl --user -u adam.service -f
```

If setup installed a unit without enabling it (for example with `--yes`), start
it with `systemctl --user enable --now adam.service`. Systems without systemd
can use foreground mode or configure their own service. Use the wake phrase
shown in your configuration, then give Adam a request.

## Optional features

### Meetings and voice profiles

Meeting mode records mixed audio and transcribes it; it can continue handling
ordinary requests while recording. The recording stops at its configured
duration limit (four hours by default). Files are saved under
`~/.local/state/adam/meetings/`; finalized microphone captures and transcript
stages are also saved under `~/.local/state/adam/heard-captures/`. These files
can contain private conversations and are created with owner-only permissions.
Say **“meeting mode on”** to start recording or **“meeting mode off”** to stop
it. Capitalization and punctuation may vary.

Optional Nemotron diarization assigns speaker turns in the mixed recording; it
does not isolate participants. Without diarization, speaker labels use voice
similarity when available and may be `Unknown`. When voice verification is
enabled, enroll with:

```sh
uv run python -m src.stt.enroll
```

### Desktop control and browser navigation

Desktop control observes the focused window before acting and refreshes the
screen between bounded actions. Its input tools depend on the active X11 or
Wayland session and installed desktop utilities. Disable it with
`computer_control.enabled: false` in `config.yaml`.

Optional browser navigation uses Playwright and Adam's separate browser
profile. It can inspect page text, navigate, follow links, fill ordinary text
fields, scroll, and use history. It does not press buttons or submit forms.
Enable it with `./setup.sh --browser-navigation` or in the setup wizard.
Treat web-page content as untrusted instructions.

Optional OmniParser adds numbered candidate control boxes to screenshots. It is
a separate detector that can use CPU or NVIDIA runtime packages; setup installs
its runtime and model when requested. Its YOLOv8 checkpoint and Ultralytics
runtime are AGPL-3.0 licensed.

Wake-free idea routing is another optional local feature. It transcribes speech
locally while Adam is idle and uses an enrolled voice profile before routing
selected requests. Enable it with `./setup.sh --idea-routing`; edit ideas in
`assets/intent_ideas.json` and thresholds in `config.yaml`.

### Web UI

The optional Web UI provides text interaction, live tool activity, and system
status in a local browser. It is disabled by default and binds to loopback. For
a foreground session, run:

```sh
uv run python -m src.main --webui
```

Then open <http://127.0.0.1:8765>. For the systemd service, set
`webui.enabled: true` in `config.yaml` and restart `adam.service`. See
[`docs/webui-sidecar-setup.md`](docs/webui-sidecar-setup.md) for configuration
and security details.

## Configuration and development

Use [`config.yaml.example`](config.yaml.example) as the reference for settings.
The local `config.yaml` contains host-specific choices and credentials. Setup
preserves it except for choices you make explicitly.

Implementation notes and evaluations are in [`docs/`](docs/). In a prepared
development environment, run the test suite with:

```sh
uv run pytest
```
