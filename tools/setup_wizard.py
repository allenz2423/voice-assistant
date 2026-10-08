#!/usr/bin/env python3
"""Comprehensive interactive setup wizard for Adam / Adam Voice Assistant.

Configures:
- Audio hardware (Microphone inputs, playback outputs)
- Assistant persona & wake word
- Speech-to-Text (STT) engine, model, language, and hardware acceleration
- Text-to-Speech (TTS) engine, voice & device
- LLM sourcing (Local Ollama, custom vLLM/llama.cpp server, or Cloud APIs: Groq/Gemini/Claude/OpenAI)
- Default web browser
- Voice profile enrollment (speaker verification)
- Optional NVIDIA Nemotron speaker diarization through Transformers
- Continuous meeting capture, transcripts, and speaker labels
- Optional local embedding idea routing for wake-free tool calls
- Optional OmniParser screenshot-region boxes for computer use
- Speech engine restored after silent mode
- Systemd background user service
"""

import getpass
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Styling helpers
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_BLUE = "\033[1;34m"
CLR_GREEN = "\033[1;32m"
CLR_YELLOW = "\033[1;33m"
CLR_CYAN = "\033[1;36m"
CLR_MAGENTA = "\033[1;35m"
CLR_RED = "\033[1;31m"

NEMOTRON_DIARIZATION_MODEL = "nvidia/Nemotron-3-Diarization"


def print_banner():
    print(f"""
{CLR_CYAN}{CLR_BOLD}================================================================
          🎙️   Adam / Adam Voice Assistant Setup
================================================================{CLR_RESET}
""")


def prompt_choice(prompt: str, options: list[str], default_idx: int = 0) -> int:
    """Prompt the user to pick an option from a numbered list."""
    for i, opt in enumerate(options, 1):
        marker = f"{CLR_GREEN}* (Current/Default){CLR_RESET}" if i - 1 == default_idx else ""
        print(f"  [{CLR_BOLD}{i}{CLR_RESET}] {opt} {marker}")

    while True:
        try:
            choice = input(f"\n{CLR_BOLD}{prompt} [1-{len(options)}] (default: {default_idx + 1}): {CLR_RESET}").strip()
            if not choice:
                return default_idx
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"{CLR_RED}Invalid option. Please choose a number between 1 and {len(options)}.{CLR_RESET}")
        except (ValueError, EOFError):
            print(f"{CLR_RED}Invalid input. Please enter a valid number.{CLR_RESET}")


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Prompt the user for a yes/no response."""
    hint = "[Y/n]" if default_yes else "[y/N]"
    while True:
        try:
            res = input(f"{CLR_BOLD}{prompt} {hint}: {CLR_RESET}").strip().lower()
            if not res:
                return default_yes
            if res in ("y", "yes"):
                return True
            if res in ("n", "no"):
                return False
            print(f"{CLR_RED}Please answer 'y' or 'n'.{CLR_RESET}")
        except EOFError:
            return default_yes


def prompt_text(prompt: str, default_val: str = "") -> str:
    """Prompt user for a freeform text string."""
    default_hint = f" (default: {default_val})" if default_val else ""
    try:
        val = input(f"{CLR_BOLD}{prompt}{default_hint}: {CLR_RESET}").strip()
        return val if val else default_val
    except EOFError:
        return default_val


def prompt_secret(prompt: str, current_value: str = "") -> str:
    """Prompts for a secret without echoing it or showing its saved value."""
    hint = " [saved key available; Enter keeps it]" if current_value else ""
    try:
        value = getpass.getpass(f"{CLR_BOLD}{prompt}{hint}: {CLR_RESET}").strip()
        return value if value else current_value
    except (EOFError, KeyboardInterrupt):
        return current_value


def update_config_value(key: str, value: str, section: str | None = None):
    """Safely updates a key: value pair in config.yaml while preserving comments."""
    config_path = PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        example_path = PROJECT_DIR / "config.yaml.example"
        if example_path.exists():
            import shutil
            shutil.copyfile(example_path, config_path)
        else:
            return

    content = config_path.read_text(encoding="utf-8")
    pattern = rf'(^[ \t]*{re.escape(key)}:\s*)(["\']?)(.*?)(["\']?)(\s*(?:#.*)?$)'
    replacement = rf'\g<1>"{value}"\g<5>'

    if section:
        section_match = re.search(rf"^{re.escape(section)}:\s*(?:#.*)?$", content, flags=re.MULTILINE)
        if not section_match:
            new_content = content.rstrip() + f'\n\n{section}:\n  {key}: "{value}"\n'
            config_path.write_text(new_content, encoding="utf-8")
            return
        next_section = re.search(r"^[^\s#][^:\n]*:\s*(?:#.*)?$", content[section_match.end():], flags=re.MULTILINE)
        section_end = section_match.end() + next_section.start() if next_section else len(content)
        section_text = content[section_match.end():section_end]
        updated_section, count = re.subn(pattern, replacement, section_text, flags=re.MULTILINE)
        if count > 0:
            config_path.write_text(content[:section_match.end()] + updated_section + content[section_end:], encoding="utf-8")
            return
        new_content = content[:section_match.end()] + f'\n  {key}: "{value}"' + content[section_match.end():]
        config_path.write_text(new_content, encoding="utf-8")
        return

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count > 0:
        config_path.write_text(new_content, encoding="utf-8")
    elif key in ("api_base", "api_key"):
        # These optional LLM settings may be missing from configs created by
        # older versions. Add them to the llm section so custom providers are
        # actually saved when the wizard is rerun.
        llm_match = re.search(r"^llm:\s*(?:#.*)?$", content, flags=re.MULTILINE)
        if llm_match:
            insert_at = llm_match.end()
            new_content = content[:insert_at] + f'\n  {key}: "{value}"' + content[insert_at:]
            config_path.write_text(new_content, encoding="utf-8")
    else:
        # If key didn't exist in config, we don't break the file
        pass


def get_current_config_value(key: str, default: str = "", section: str | None = None) -> str:
    """Reads a current configuration string value from config.yaml or config.yaml.example."""
    config_path = PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_DIR / "config.yaml.example"
    if not config_path.exists():
        return default
    content = config_path.read_text(encoding="utf-8")
    if section:
        section_match = re.search(rf"^{re.escape(section)}:\s*(?:#.*)?$", content, flags=re.MULTILINE)
        if not section_match:
            return default
        next_section = re.search(r"^[^\s#][^:\n]*:\s*(?:#.*)?$", content[section_match.end():], flags=re.MULTILINE)
        section_end = section_match.end() + next_section.start() if next_section else len(content)
        content = content[section_match.end():section_end]
    match = re.search(rf'^[ \t]*{re.escape(key)}:\s*["\']?(.*?)["\']?\s*(?:#.*)?$', content, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def query_pulse_devices(kind: str = "sources") -> list[dict[str, str]]:
    """Queries pactl for list of sources or sinks."""
    devices = []
    try:
        out = subprocess.check_output(["pactl", "list", kind], text=True, stderr=subprocess.DEVNULL)
        curr = {}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(f"{kind[:-1].capitalize()} #") or line.startswith("Source #") or line.startswith("Sink #"):
                if "name" in curr:
                    devices.append(curr)
                curr = {}
            elif line.startswith("Name: "):
                curr["name"] = line.split("Name: ", 1)[1].strip()
            elif line.startswith("Description: "):
                curr["desc"] = line.split("Description: ", 1)[1].strip()
        if "name" in curr:
            devices.append(curr)
    except Exception:
        pass
    return devices


# ------------------------------------------------------------------------------
# Step 1: Audio Hardware
# ------------------------------------------------------------------------------
def configure_audio():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 1: Audio Hardware Configuration ---{CLR_RESET}")

    all_sources = query_pulse_devices("sources")
    mics = [s for s in all_sources if not s.get("name", "").endswith(".monitor")]
    current_source = get_current_config_value("target_source")

    if mics:
        print(f"\n{CLR_BOLD}Detected Microphones (Audio Inputs):{CLR_RESET}")
        options = [f"{m.get('desc', m['name'])} ({m['name']})" for m in mics]
        default_idx = 0
        for i, m in enumerate(mics):
            if m["name"] == current_source:
                default_idx = i
                break

        chosen_idx = prompt_choice("Select your primary microphone", options, default_idx)
        chosen_mic = mics[chosen_idx]["name"]
        update_config_value("target_source", chosen_mic)
        print(f"{CLR_GREEN}Configured microphone:{CLR_RESET} {chosen_mic}")
    else:
        print(f"{CLR_YELLOW}No dedicated microphones detected via pactl; keeping: {current_source}{CLR_RESET}")

    sinks = query_pulse_devices("sinks")
    current_sink = get_current_config_value("target_sink")
    if sinks:
        print(f"\n{CLR_BOLD}Detected Speakers & Headphones (Audio Outputs):{CLR_RESET}")
        options = [f"{s.get('desc', s['name'])} ({s['name']})" for s in sinks]
        default_idx = 0
        for i, s in enumerate(sinks):
            if s["name"] == current_sink:
                default_idx = i
                break

        chosen_idx = prompt_choice("Select your primary audio playback output", options, default_idx)
        chosen_sink = sinks[chosen_idx]["name"]
        update_config_value("target_sink", chosen_sink)
        print(f"{CLR_GREEN}Configured output sink:{CLR_RESET} {chosen_sink}")
    else:
        print(f"{CLR_YELLOW}No output sinks detected via pactl; keeping: {current_sink}{CLR_RESET}")

    if prompt_yes_no("\nWould you like to run a 4-second microphone & transcription test?", default_yes=False):
        test_script = PROJECT_DIR / "tools" / "test_mic.py"
        if test_script.exists():
            subprocess.run([sys.executable, str(test_script)])


# ------------------------------------------------------------------------------
# Step 2: Wake Word & Persona
# ------------------------------------------------------------------------------
def configure_persona():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 2: Wake Word & Persona ---{CLR_RESET}")
    current_wake = get_current_config_value("wake_word", "hey adam")

    personas = [
        "Hey Adam  - Warm, studio male voice (Kokoro am_adam)",
        "Hey Jarvis - Classic AI assistant voice (Kokoro am_michael)",
        "Custom wake word phrase",
    ]

    default_idx = 0
    if "jarvis" in current_wake.lower():
        default_idx = 1
    elif "adam" not in current_wake.lower():
        default_idx = 2

    choice = prompt_choice("Choose assistant wake word", personas, default_idx)
    if choice == 0:
        update_config_value("wake_word", "hey adam")
        update_config_value("voice", "am_adam")
        print(f"{CLR_GREEN}Wake word set to 'hey adam'.{CLR_RESET}")
    elif choice == 1:
        update_config_value("wake_word", "hey jarvis")
        update_config_value("voice", "am_michael")
        print(f"{CLR_GREEN}Wake word set to 'hey jarvis'.{CLR_RESET}")
    else:
        custom_wake = prompt_text("Enter custom wake word phrase", current_wake)
        update_config_value("wake_word", custom_wake)
        print(f"{CLR_GREEN}Wake word updated to '{custom_wake}'.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 3: Speech-to-Text (STT) Model & Hardware
# ------------------------------------------------------------------------------
def detect_nvidia_devices() -> list[tuple[str, str]]:
    """Return physical CUDA indices and names without selecting a GPU implicitly."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0].isdigit():
            devices.append((parts[0], parts[1]))
    return devices


def detect_qwen_vulkan_devices() -> list[tuple[str, str]]:
    """List Vulkan GPU backends Qwen can use; do not default to another GPU."""
    try:
        import transcribe_cpp
        return [
            (str(device.name), str(device.description))
            for device in transcribe_cpp.backends()
            if "vulkan" in str(getattr(device, "kind", "")).lower()
            and str(getattr(device, "device_type", "")).lower() == "gpu"
        ]
    except (ImportError, OSError, RuntimeError):
        return []


def choose_nvidia_device(prompt: str, devices: list[tuple[str, str]], default_index: str = "0") -> str:
    if devices:
        labels = [f"CUDA:{index} — {name}" for index, name in devices]
        selected = next((i for i, (index, _) in enumerate(devices) if index == default_index), 0)
        return devices[prompt_choice(prompt, labels, selected)][0]
    print(f"{CLR_YELLOW}No NVIDIA GPU was detected with nvidia-smi; CUDA may fail and fall back to CPU.{CLR_RESET}")
    value = prompt_text("CUDA GPU index", default_index)
    while not value.isdigit():
        print(f"{CLR_RED}Enter a non-negative GPU index, such as 0.{CLR_RESET}")
        value = prompt_text("CUDA GPU index", default_index)
    return value


def ensure_nvidia_runtime() -> bool:
    """Install NVIDIA ONNX/CUDA wheels only when a GPU feature is selected."""
    if os.environ.get("ADAM_RUNTIME_EXTRA", "runtime-cpu") == "runtime-nvidia":
        return True
    if os.environ.get("ADAM_CPU_ONLY") == "1":
        print(f"{CLR_YELLOW}CUDA support was disabled by --cpu-only.{CLR_RESET}")
        return False
    if os.environ.get("ADAM_SKIP_PYTHON") == "1":
        print(f"{CLR_YELLOW}Cannot install CUDA support because Python dependency installation was skipped.{CLR_RESET}")
        return False
    if not prompt_yes_no("Install NVIDIA CUDA runtime support for the selected GPU?", default_yes=True):
        return False
    uv = shutil.which("uv")
    if not uv:
        print(f"{CLR_RED}uv was not found; install the NVIDIA runtime with './setup.sh --nvidia-runtime'.{CLR_RESET}")
        return False
    args = [uv, "sync", "--extra", "runtime-nvidia"]
    if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") != "1":
        args.extend(["--extra", "speaker-verification"])
    if os.environ.get("ADAM_DIARIZATION_ENABLED") == "1":
        args.extend(["--extra", "nemotron-diarization"])
    try:
        subprocess.run(args, cwd=PROJECT_DIR, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{CLR_RED}Could not install NVIDIA runtime ({exc}); leaving the CPU runtime active.{CLR_RESET}")
        return False
    os.environ["ADAM_RUNTIME_EXTRA"] = "runtime-nvidia"
    print(f"{CLR_GREEN}NVIDIA runtime installed. Adam will use only the GPU you select.{CLR_RESET}")
    return True


def configure_stt():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 3: Speech-to-Text (STT) Engine & Hardware ---{CLR_RESET}")
    current_provider = get_current_config_value("provider", "local", section="stt")
    current_model = get_current_config_value("model_size", "qwen3-asr-1.7b", section="stt")
    current_dev = get_current_config_value("device", "Vulkan0", section="stt")
    nvidia_devices = [] if os.environ.get("ADAM_CPU_ONLY") == "1" else detect_nvidia_devices()
    qwen_devices = [] if os.environ.get("ADAM_CPU_ONLY") == "1" else detect_qwen_vulkan_devices()

    stt_options = [
        "Qwen3-ASR-1.7B on a selected Vulkan GPU (desktop, fast and accurate)",
        "Faster-Whisper small.en on a selected NVIDIA CUDA GPU",
        "Faster-Whisper small.en on CPU (laptop-friendly, no GPU required)",
        "Custom STT model & device",
        "OpenAI Cloud transcription (audio is sent to OpenAI)",
        "OpenRouter Cloud transcription (choose any available STT model)",
        "Custom cloud transcription endpoint + model (OpenAI-compatible)",
    ]

    default_idx = 0 if qwen_devices else 2
    if "small.en" in current_model.lower():
        default_idx = 1 if current_dev.lower() == "cuda" and nvidia_devices else 2
    elif "base" in current_model.lower():
        default_idx = 2 if current_dev.lower() == "cpu" else 3
    elif current_provider != "local":
        default_idx = 4 if current_provider == "openai" else 5 if current_provider == "openrouter" else 6
    elif "qwen" in current_model.lower() and qwen_devices:
        default_idx = 0
    elif "qwen" in current_model.lower() and not qwen_devices:
        default_idx = 2
    elif "small.en" not in current_model.lower() and "base" not in current_model.lower():
        default_idx = 3

    choice = prompt_choice("Select STT engine configuration", stt_options, default_idx)

    if choice == 0:
        if os.environ.get("ADAM_CPU_ONLY") == "1":
            print(f"{CLR_YELLOW}Qwen GPU inference is unavailable in CPU-only mode. Select small.en on CPU instead.{CLR_RESET}")
            return
        if qwen_devices:
            labels = [f"{name} — {description}" for name, description in qwen_devices]
            default_device = next((i for i, (name, _) in enumerate(qwen_devices) if name.lower() == current_dev.lower()), 0)
            device = qwen_devices[prompt_choice("Select the Vulkan GPU for Qwen", labels, default_device)][0]
        else:
            print(f"{CLR_YELLOW}No Vulkan GPU was detected. Qwen may fall back to CPU and run slowly.{CLR_RESET}")
            device = prompt_text("Qwen backend name (for example Vulkan0, Vulkan1, or CPU)", current_dev)
        update_config_value("provider", "local", section="stt")
        update_config_value("model_size", "qwen3-asr-1.7b", section="stt")
        update_config_value("device", device, section="stt")
        update_config_value("compute_type", "int8_float32", section="stt")
        update_config_value("language", "", section="stt")
        print(f"{CLR_GREEN}Configured Qwen3-ASR on {device}; it will detect the spoken language automatically.{CLR_RESET}")
    elif choice == 1:
        if not ensure_nvidia_runtime():
            print(f"{CLR_YELLOW}Keeping the current STT device because CUDA runtime support is unavailable.{CLR_RESET}")
            return
        current_index = str(get_current_config_value("device_index", "0", section="stt"))
        device_index = choose_nvidia_device("Select the CUDA GPU for Whisper", nvidia_devices, current_index)
        precision_options = [
            "INT8 with FP32 compute (compatible with older NVIDIA GPUs)",
            "FP16 (faster on newer NVIDIA GPUs)",
        ]
        precision_default = 1 if get_current_config_value("compute_type", "int8_float32", section="stt") == "float16" else 0
        compute_type = ("int8_float32", "float16")[prompt_choice("Select GPU precision", precision_options, precision_default)]
        update_config_value("provider", "local", section="stt")
        update_config_value("model_size", "small.en", section="stt")
        update_config_value("device", "cuda", section="stt")
        update_config_value("device_index", device_index, section="stt")
        update_config_value("compute_type", compute_type, section="stt")
        print(f"{CLR_GREEN}Configured faster-whisper small.en on CUDA:{device_index} ({compute_type}).{CLR_RESET}")
    elif choice == 2:
        update_config_value("provider", "local", section="stt")
        update_config_value("model_size", "small.en", section="stt")
        update_config_value("device", "cpu", section="stt")
        update_config_value("compute_type", "int8", section="stt")
        print(f"{CLR_GREEN}Configured faster-whisper small.en on CPU.{CLR_RESET}")
    elif choice == 3:
        custom_model = prompt_text("Enter STT model name (for example qwen3-asr-1.7b, small.en, or medium.en)", current_model)
        custom_dev = prompt_text("Enter compute device (cpu, cuda, or the exact Vulkan backend name)", current_dev)
        update_config_value("provider", "local", section="stt")
        update_config_value("model_size", custom_model, section="stt")
        update_config_value("device", custom_dev, section="stt")
        if custom_dev.lower() == "cuda":
            if not ensure_nvidia_runtime():
                print(f"{CLR_YELLOW}Keeping the existing local STT configuration.{CLR_RESET}")
                return
            current_index = str(get_current_config_value("device_index", "0", section="stt"))
            device_index = choose_nvidia_device("Select the CUDA GPU", nvidia_devices, current_index)
            update_config_value("device_index", device_index, section="stt")
            compute_type = prompt_text(
                "CTranslate2 compute type (float16, int8_float16, int8_float32)",
                get_current_config_value("compute_type", "int8_float32", section="stt"),
            )
            update_config_value("compute_type", compute_type, section="stt")
        if "qwen" in custom_model.lower():
            update_config_value("language", "", section="stt")
            print(f"{CLR_CYAN}Qwen3-ASR will detect the spoken language automatically; this runtime does not accept language hints.{CLR_RESET}")
        print(f"{CLR_GREEN}Configured custom STT:{CLR_RESET} model={custom_model}, device={custom_dev}")
    else:
        is_openrouter = choice == 5
        is_custom = choice == 6
        cloud_provider = "custom" if is_custom else "openrouter" if is_openrouter else "openai"
        update_config_value("provider", cloud_provider, section="stt")
        if is_custom:
            default_url = get_current_config_value("cloud_url", "", section="stt")
            cloud_url = prompt_text(
                "Cloud transcription URL (OpenAI-compatible; e.g. https://api.example.com/v1/audio/transcriptions)",
                default_url,
            )
            update_config_value("cloud_url", cloud_url, section="stt")
        elif is_openrouter:
            update_config_value("cloud_url", "https://openrouter.ai/api/v1/audio/transcriptions", section="stt")
        else:
            update_config_value("cloud_url", "https://api.openai.com/v1/audio/transcriptions", section="stt")

        default_model = get_current_config_value("cloud_model", "gpt-transcribe", section="stt")
        if is_openrouter and current_provider != "openrouter":
            default_model = "openai/whisper-large-v3-turbo"
        elif not is_openrouter and not is_custom and current_provider != "openai":
            default_model = "gpt-transcribe"
        model = prompt_text("Cloud transcription model name", default_model)
        update_config_value("cloud_model", model, section="stt")
        saved_key = get_current_config_value("api_key", "", section="stt")
        llm_provider = get_current_config_value("provider", "local", section="llm")
        llm_endpoint = get_current_config_value("api_base", "", section="llm").lower()
        shared_key = ""
        if is_openrouter:
            shared_key = get_current_config_value("api_key", "", section="llm") if "openrouter.ai" in llm_endpoint else ""
            saved_key = saved_key or os.environ.get("OPENROUTER_API_KEY", "") or shared_key
            key_prompt = "OpenRouter API key"
        elif is_custom:
            shared_key = get_current_config_value("api_key", "", section="llm") if llm_provider in ("custom", "openai_compatible") else ""
            saved_key = saved_key or shared_key
            key_prompt = "Cloud API key / Bearer token"
        else:
            saved_key = saved_key or os.environ.get("OPENAI_API_KEY", "")
            key_prompt = "OpenAI API key"
        api_key = prompt_secret(key_prompt, saved_key)
        if api_key and api_key != shared_key:
            update_config_value("api_key", api_key, section="stt")

        fallback_model = prompt_text(
            "Faster-Whisper fallback model",
            get_current_config_value("fallback_model", "base.en", section="stt"),
        )
        fallback_device = prompt_text(
            "Faster-Whisper fallback device (cpu or cuda)",
            get_current_config_value("fallback_device", "cpu", section="stt"),
        )
        update_config_value("fallback_model", fallback_model, section="stt")
        update_config_value("fallback_device", fallback_device, section="stt")
        update_config_value(
            "fallback_compute_type",
            "int8_float32" if fallback_device.lower() == "cuda" else "int8",
            section="stt",
        )
        provider_label = "custom cloud" if is_custom else "OpenRouter cloud" if is_openrouter else "OpenAI cloud"
        print(f"{CLR_GREEN}Configured {provider_label} transcription ({model}) with Faster-Whisper fallback ({fallback_model} on {fallback_device}).{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 4: Text-to-Speech (TTS) Engine & Voice
# ------------------------------------------------------------------------------
def configure_tts():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 4: Text-to-Speech (TTS) Engine & Voice ---{CLR_RESET}")
    current_engine = get_current_config_value("engine", "kokoro")
    current_voice = get_current_config_value("voice", "am_adam")
    restore_engine = get_current_config_value("silent_restore_engine", "kokoro", section="tts").lower()

    engines = [
        "Kokoro-82M Neural TTS (High fidelity, natural phrasing)",
        "OpenAI Cloud TTS (audio is sent to OpenAI)",
        "Silent (show responses as desktop notification alerts)",
    ]
    engine_idx = {"openai": 1, "silent": 2}.get(current_engine, 0)
    chosen_engine_idx = prompt_choice("Select TTS engine", engines, engine_idx)

    if chosen_engine_idx == 0:
        update_config_value("engine", "kokoro", section="tts")
        nvidia_devices = [] if os.environ.get("ADAM_CPU_ONLY") == "1" else detect_nvidia_devices()
        current_device_id = get_current_config_value("device_id", "0", section="tts")
        device_options = ["CPU (works on laptops without CUDA)"]
        if nvidia_devices:
            device_options.extend([f"CUDA:{index} — {name}" for index, name in nvidia_devices])
            default_device_idx = next(
                (i + 1 for i, (index, _) in enumerate(nvidia_devices) if index == current_device_id),
                0,
            )
        else:
            default_device_idx = 0
        device_choice = prompt_choice("Select Kokoro compute device", device_options, default_device_idx)
        if device_choice > 0 and not ensure_nvidia_runtime():
            print(f"{CLR_YELLOW}Kokoro will use CPU because CUDA support is unavailable.{CLR_RESET}")
            device_choice = 0
        if device_choice == 0:
            update_config_value("device_id", "-1", section="tts")
        else:
            selected_index = nvidia_devices[device_choice - 1][0]
            update_config_value("device_id", selected_index, section="tts")
        print(f"{CLR_GREEN}Kokoro device set to {'CPU' if device_choice == 0 else 'CUDA:' + selected_index}.{CLR_RESET}")
        voices = [
            "am_adam   - Studio natural male voice",
            "am_michael - Authoritative assistant tone",
            "af_bella  - Clear, melodic female voice",
            "af_nicole - Soft spoken female voice",
            "Custom Kokoro voice code",
        ]
        voice_default = 0
        if "michael" in current_voice:
            voice_default = 1
        elif "bella" in current_voice:
            voice_default = 2
        elif "nicole" in current_voice:
            voice_default = 3

        v_choice = prompt_choice("Select voice personality", voices, voice_default)
        voice_map = {0: "am_adam", 1: "am_michael", 2: "af_bella", 3: "af_nicole"}
        if v_choice in voice_map:
            update_config_value("voice", voice_map[v_choice], section="tts")
            print(f"{CLR_GREEN}Configured voice:{CLR_RESET} {voice_map[v_choice]}")
        else:
            custom_v = prompt_text("Enter voice code (e.g. 'af_sarah', 'am_fenrir')", current_voice)
            update_config_value("voice", custom_v, section="tts")
            print(f"{CLR_GREEN}Configured custom voice:{CLR_RESET} {custom_v}")
    elif chosen_engine_idx == 1:
        update_config_value("engine", "openai", section="tts")
        model = prompt_text("OpenAI speech model", get_current_config_value("cloud_model", "gpt-4o-mini-tts", section="tts"))
        update_config_value("cloud_model", model, section="tts")
        voice = prompt_text("OpenAI voice (for example marin, cedar, coral, alloy)", get_current_config_value("cloud_voice", "marin", section="tts"))
        update_config_value("cloud_voice", voice, section="tts")
        api_key = prompt_secret("OpenAI API key", get_current_config_value("api_key", "", section="tts") or os.environ.get("OPENAI_API_KEY", ""))
        if api_key:
            update_config_value("api_key", api_key, section="tts")
        print(f"{CLR_GREEN}Configured OpenAI cloud speech ({model}, voice {voice}).{CLR_RESET}")
    else:
        update_config_value("engine", "silent", section="tts")
        restore_options = [
            "Kokoro (local speech)",
            "OpenAI (cloud speech)",
        ]
        restore_idx = 1 if restore_engine == "openai" else 0
        restore_choice = prompt_choice("Choose the speech engine restored when silent mode is disabled", restore_options, restore_idx)
        restore_engine = "openai" if restore_choice == 1 else "kokoro"
        print(f"{CLR_GREEN}Configured silent mode; responses will appear as desktop notification alerts.{CLR_RESET}")

    if chosen_engine_idx != 2:
        restore_engine = "openai" if chosen_engine_idx == 1 else "kokoro"
    update_config_value("silent_restore_engine", restore_engine, section="tts")
    print(f"{CLR_GREEN}Silent mode will restore {restore_engine} speech when disabled.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 5: LLM Sourcing & Models (Local vs Cloud, Endpoints, Models)
# ------------------------------------------------------------------------------
def configure_llm():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 5: LLM Brain Sourcing & Model Configuration ---{CLR_RESET}")
    current_provider = get_current_config_value("provider", "local", section="llm")

    source_options = [
        "Local LLM (Run locally on your machine / home server via Ollama or OpenAI-compatible server)",
        "Cloud LLM API (Ultra-fast cloud inference: Groq, Gemini, Anthropic Claude, OpenAI)",
    ]
    default_source = 0 if current_provider in ("local", "custom", "openai_compatible", "vllm") else 1
    source_choice = prompt_choice("Where should Adam source the LLM brain?", source_options, default_source)

    if source_choice == 0:
        # Local Setup
        local_backends = [
            "Ollama (Default: http://localhost:11434)",
            "Custom / Remote OpenAI-compatible endpoint (vLLM, llama.cpp server, LM Studio, LocalAI, or remote Ollama)",
        ]
        curr_endpoint = get_current_config_value("ollama_host", "http://localhost:11434", section="llm")
        default_backend = 0 if "11434" in curr_endpoint and current_provider == "local" else 1

        b_choice = prompt_choice("Select local runtime backend", local_backends, default_backend)

        if b_choice == 0:
            update_config_value("provider", "local", section="llm")
            update_config_value("ollama_host", "http://localhost:11434", section="llm")
            endpoint = "http://localhost:11434"
        else:
            update_config_value("provider", "openai_compatible", section="llm")
            endpoint = prompt_text("Enter local/remote server endpoint URL", curr_endpoint if curr_endpoint else "http://localhost:8000/v1")
            update_config_value("ollama_host", endpoint, section="llm")
            update_config_value("api_base", endpoint, section="llm")
            api_key = prompt_secret("Optional API key / Bearer token (leave blank if none)", get_current_config_value("api_key", "", section="llm"))
            if api_key:
                update_config_value("api_key", api_key, section="llm")

        # Select model
        curr_model = get_current_config_value("local_model", "qwen3.5:4b", section="llm")
        models = [
            "qwen3.5:4b   - (Recommended) Fast 4B parameter model with 64k context",
            "qwen2.5:7b   - Powerful reasoning & structured JSON calling (~4.5GB VRAM)",
            "llama3.2:3b  - Extremely lightweight & responsive (~2.0GB VRAM)",
            "Custom local model tag (type your own)",
        ]
        default_m = 0
        if "7b" in curr_model:
            default_m = 1
        elif "3b" in curr_model or "llama" in curr_model:
            default_m = 2
        elif "4b" not in curr_model:
            default_m = 3

        m_choice = prompt_choice("Select local model", models, default_m)
        m_map = {0: "qwen3.5:4b", 1: "qwen2.5:7b", 2: "llama3.2:3b"}
        if m_choice in m_map:
            target_model = m_map[m_choice]
        else:
            target_model = prompt_text("Enter model name (e.g. 'mistral', 'deepseek-r1:8b')", curr_model)

        update_config_value("local_model", target_model, section="llm")
        print(f"{CLR_GREEN}Local model configured:{CLR_RESET} {target_model} at {endpoint}")

        # If standard local Ollama is available, offer to pull
        if os.environ.get("ADAM_SKIP_OLLAMA") == "1":
            print(f"{CLR_YELLOW}Ollama model checks and downloads skipped by setup option.{CLR_RESET}")
        elif b_choice == 0 and subprocess.run(["command", "-v", "ollama"], shell=True, stdout=subprocess.DEVNULL).returncode == 0:
            if prompt_yes_no(f"Pull '{target_model}' into Ollama now?", default_yes=True):
                print(f"{CLR_CYAN}Pulling {target_model}...{CLR_RESET}")
                subprocess.run(["ollama", "pull", target_model])

    else:
        # Cloud API Setup
        cloud_providers = [
            "Groq           - Near-instant inference (LLaMA-3.3-70B, llama-3.1-8b)",
            "Google Gemini  - Fast & reliable (gemini-2.5-flash)",
            "Anthropic      - Highest reasoning accuracy (claude-3-5-sonnet)",
            "OpenAI         - Industry standard (gpt-4o-mini, gpt-4o)",
            "Custom         - Your own OpenAI-compatible provider / endpoint",
        ]
        p_choice = prompt_choice("Select cloud provider", cloud_providers, 0)
        if p_choice == 4:
            update_config_value("provider", "custom", section="llm")
            current_base = get_current_config_value("api_base", "", section="llm")
            api_base = prompt_text("Enter API base URL (for example https://api.example.com/v1)", current_base)
            update_config_value("api_base", api_base, section="llm")

            current_model = get_current_config_value("cloud_model", "", section="llm") or get_current_config_value("local_model", "", section="llm")
            model = prompt_text("Enter provider model name", current_model)
            update_config_value("cloud_model", model, section="llm")

            api_key = prompt_secret("Enter API key / Bearer token (leave blank if none)", get_current_config_value("api_key", "", section="llm"))
            if api_key:
                update_config_value("api_key", api_key, section="llm")
            print(f"{CLR_GREEN}Custom OpenAI-compatible provider configured for model:{CLR_RESET} {model}")
            return

        p_keys = {0: "groq", 1: "gemini", 2: "anthropic", 3: "openai"}
        provider_name = p_keys[p_choice]
        update_config_value("provider", provider_name, section="llm")

        model_defaults = {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini"
        }
        chosen_cloud_model = prompt_text(f"Enter {provider_name.capitalize()} model name", model_defaults[provider_name])
        update_config_value("cloud_model", chosen_cloud_model, section="llm")

        key_env_var = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }[provider_name]

        existing_key = os.environ.get(key_env_var, get_current_config_value("api_key", "", section="llm"))
        api_key = prompt_secret(f"Enter {key_env_var}", existing_key)
        if api_key:
            update_config_value("api_key", api_key, section="llm")
            print(f"{CLR_GREEN}API key configured for {provider_name}.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 6: Default Web Browser
# ------------------------------------------------------------------------------
def configure_browser():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 6: Default Web Browser ---{CLR_RESET}")
    current_browser = get_current_config_value("default_browser", "microsoft-edge-stable")

    browsers = [
        "Microsoft Edge (microsoft-edge-stable)",
        "Mozilla Firefox (firefox)",
        "Google Chrome (google-chrome-stable)",
        "Brave Browser (brave)",
        "Zen Browser (zen-browser)",
        "Chromium (chromium)",
        "Vivaldi (vivaldi-stable)",
        "Custom browser command",
    ]

    browser_binaries = [
        "microsoft-edge-stable", "firefox", "google-chrome-stable", "brave",
        "zen-browser", "chromium", "vivaldi-stable",
    ]
    default_idx = next(
        (i for i, binary in enumerate(browser_binaries) if binary in current_browser and shutil.which(binary)),
        next((i for i, binary in enumerate(browser_binaries) if shutil.which(binary)), 7),
    )

    choice = prompt_choice("Choose default browser for web searches and links", browsers, default_idx)
    browser_map = {
        0: "microsoft-edge-stable",
        1: "firefox",
        2: "google-chrome-stable",
        3: "brave",
        4: "zen-browser",
        5: "chromium",
        6: "vivaldi-stable",
    }

    if choice in browser_map:
        update_config_value("default_browser", browser_map[choice])
        print(f"{CLR_GREEN}Default browser set to:{CLR_RESET} {browser_map[choice]}")
    else:
        custom_b = prompt_text("Enter browser binary name (e.g. 'chromium', 'floorp')", current_browser)
        update_config_value("default_browser", custom_b)
        print(f"{CLR_GREEN}Default browser set to:{CLR_RESET} {custom_b}")


# ------------------------------------------------------------------------------
# Step 7: Speaker Verification Enrollment
# ------------------------------------------------------------------------------
def configure_speaker_verification():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 7: Speaker Verification (Voice Profile) ---{CLR_RESET}")
    currently_enabled = get_current_config_value("enabled", "true", section="speaker_verification").lower() in ("true", "yes", "1")
    if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") == "1":
        update_config_value("enabled", "false", section="speaker_verification")
        print(f"{CLR_YELLOW}Speaker verification skipped by setup option.{CLR_RESET}")
        return
    if not prompt_yes_no("Enable speaker verification and identify you in meetings?", default_yes=currently_enabled):
        update_config_value("enabled", "false", section="speaker_verification")
        print(f"{CLR_YELLOW}Speaker verification is disabled. Meeting labels will remain anonymous.{CLR_RESET}")
        return
    update_config_value("enabled", "true", section="speaker_verification")
    profile_path = Path.home() / ".local" / "state" / "adam" / "speaker-profile.npz"

    if profile_path.exists():
        print(f"{CLR_GREEN}Existing voice profile detected at:{CLR_RESET} {profile_path}")
        enroll_prompt = "Would you like to re-enroll and replace your voice profile now?"
        default_enroll = False
    else:
        print("Speaker verification ensures the assistant only listens to your unique voice.")
        enroll_prompt = "Would you like to enroll your voice profile now (takes ~30s)?"
        default_enroll = True

    if os.environ.get("ADAM_SKIP_ENROLLMENT") != "1" and prompt_yes_no(enroll_prompt, default_yes=default_enroll):
        print(f"\n{CLR_CYAN}Starting enrollment session... Speak clearly when prompted.{CLR_RESET}")
        try:
            from src.stt.enroll import main as run_enroll
            run_enroll()
        except Exception as e:
            print(f"{CLR_RED}Enrollment encountered an error: {e}{CLR_RESET}")
            print(f"You can enroll anytime later by running: {CLR_BOLD}uv run python -m src.stt.enroll{CLR_RESET}")


def configure_meeting():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 8: Meeting Mode & Recording ---{CLR_RESET}")
    print(
        "Say 'Hey Adam, meeting mode' to record and transcribe continuously. "
        "Meeting audio is kept intact; diarization labels who spoke without filtering anyone out."
    )
    print(
        f"Every finalized microphone capture and transcript is also saved privately under "
        f"{Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'adam' / 'heard-captures'}."
    )
    default_dir = get_current_config_value("output_dir", "~/.local/state/adam/meetings", section="meeting")
    output_dir = prompt_text("Meeting recordings and transcripts directory", default_dir)
    update_config_value("output_dir", output_dir, section="meeting")

    max_hours = prompt_text(
        "Maximum meeting duration in hours (automatically stops at the limit)",
        get_current_config_value("max_duration_hours", "4.0", section="meeting"),
    )
    while True:
        try:
            if 0.25 <= float(max_hours) <= 24:
                break
        except ValueError:
            pass
        print(f"{CLR_RED}Enter a duration from 0.25 to 24 hours.{CLR_RESET}")
        max_hours = prompt_text("Maximum meeting duration in hours", "4.0")
    update_config_value("max_duration_hours", max_hours, section="meeting")
    print(f"{CLR_GREEN}Meeting recording is configured for up to {max_hours} hours.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 8: Optional Speaker Diarization
# ------------------------------------------------------------------------------
def configure_speaker_diarization():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 9: Optional Nemotron Speaker Diarization ---{CLR_RESET}")
    print(
        "Diarization assigns separate speaker labels to the intact meeting transcript. "
        "It does not remove, isolate, or filter any speaker's audio. "
        "Choose CPU or an explicit NVIDIA CUDA device; Vulkan is not supported here."
    )

    current_enabled = get_current_config_value("enabled", "false", section="speaker_diarization").lower() in ("true", "yes", "1")
    if not prompt_yes_no("Enable optional Nemotron diarization?", default_yes=current_enabled):
        update_config_value("enabled", "false", section="speaker_diarization")
        print(f"{CLR_YELLOW}Nemotron diarization remains disabled.{CLR_RESET}")
        return

    current_model = get_current_config_value("model", NEMOTRON_DIARIZATION_MODEL, section="speaker_diarization")
    if current_model.strip().lower().endswith(".gguf"):
        current_model = NEMOTRON_DIARIZATION_MODEL
    model = prompt_text("Diarization model", current_model)
    current_device = get_current_config_value("device", "auto", section="speaker_diarization").strip().lower()
    current_backend = "cpu" if current_device == "cpu" else "cuda" if current_device.startswith("cuda") else "auto"
    nvidia_devices = [] if os.environ.get("ADAM_CPU_ONLY") == "1" else detect_nvidia_devices()
    device_options = ["CPU (recommended for laptops; does not select a GPU)"]
    device_options.extend([f"CUDA:{index} — {name}" for index, name in nvidia_devices])
    backend_idx = 0 if current_backend == "cpu" or not nvidia_devices else next(
        (i + 1 for i, (index, _) in enumerate(nvidia_devices) if current_device == f"cuda:{index}"), 0
    )
    backend_choice = prompt_choice("Choose compute device", device_options, backend_idx)
    if backend_choice == 0:
        device = "cpu"
    else:
        device = f"cuda:{nvidia_devices[backend_choice - 1][0]}"
    update_config_value("enabled", "true", section="speaker_diarization")
    update_config_value("executable", "nemo-speech", section="speaker_diarization")
    update_config_value("model", model, section="speaker_diarization")
    update_config_value("device", device, section="speaker_diarization")
    os.environ["ADAM_DIARIZATION_ENABLED"] = "1"

    if os.environ.get("ADAM_SKIP_PYTHON") == "1":
        import importlib.util
        if not all(importlib.util.find_spec(module) for module in ("transformers", "torch", "librosa")):
            print(f"{CLR_YELLOW}Diarization dependencies are missing and --skip-python-deps prevents installing them.{CLR_RESET}")
            update_config_value("enabled", "false", section="speaker_diarization")
            return
        print(f"{CLR_GREEN}Using the already-installed diarization dependencies.{CLR_RESET}")

    uv = shutil.which("uv")
    if not uv:
        print(f"{CLR_RED}uv was not found; install project dependencies before enabling diarization.{CLR_RESET}")
        update_config_value("enabled", "false", section="speaker_diarization")
        return
    try:
        if os.environ.get("ADAM_SKIP_PYTHON") != "1":
            sync_args = [uv, "sync", "--extra", os.environ.get("ADAM_RUNTIME_EXTRA", "runtime-cpu"), "--extra", "nemotron-diarization"]
            if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") != "1":
                sync_args.extend(["--extra", "speaker-verification"])
            subprocess.run(sync_args, cwd=PROJECT_DIR, check=True)
        if Path(model).expanduser().is_dir():
            print(f"{CLR_GREEN}Using local Nemotron model directory: {model}{CLR_RESET}")
        elif prompt_yes_no("Download the Nemotron model now (~400 MB)?", default_yes=True):
            subprocess.run(
                [uv, "run", "python", "-c",
                 "import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], allow_patterns=['config.json', 'model.safetensors', 'processor_config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json'])",
                 model],
                cwd=PROJECT_DIR,
                check=True,
            )
        print(f"{CLR_GREEN}Nemotron diarization enabled. The model will load on first use.{CLR_RESET}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{CLR_YELLOW}Diarization setup failed ({exc}); setting was saved but diarization is inactive until setup succeeds.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 10: Optional Idea-Based Wake-Free Routing
# ------------------------------------------------------------------------------
def configure_idea_routing():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 10: Optional Idea-Based Wake-Free Routing ---{CLR_RESET}")
    print(
        "Adam can use a small local MiniLM embedding model to match idle speech to configured ideas. "
        "Direct command matches enter the normal assistant flow; calendar ideas get an isolated LLM review."
    )
    print(
        "This transcribes detected speech locally while idle, requires an enrolled voice profile, "
        "and is paused during meeting mode. No audio is sent to cloud transcription for this feature."
    )
    current_enabled = get_current_config_value("enabled", "false", section="idea_routing").lower() in ("true", "yes", "1")
    default_enabled = os.environ.get("ADAM_IDEA_ROUTING_DEFAULT") == "1" or current_enabled
    if not prompt_yes_no("Enable wake-free command matching and background calendar review?", default_yes=default_enabled):
        update_config_value("enabled", "false", section="idea_routing")
        print(f"{CLR_YELLOW}Idea-based routing remains disabled.{CLR_RESET}")
        return

    if os.environ.get("ADAM_SKIP_ENROLLMENT") == "1":
        print(f"{CLR_YELLOW}Idea routing needs a voice profile, but --skip-enrollment was selected. Leaving it disabled.{CLR_RESET}")
        update_config_value("enabled", "false", section="idea_routing")
        return

    if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") == "1":
        print(f"{CLR_YELLOW}Idea routing requires speaker verification, but --skip-speaker-verification was selected. Leaving it disabled.{CLR_RESET}")
        update_config_value("enabled", "false", section="idea_routing")
        return

    speaker_enabled = get_current_config_value("enabled", "true", section="speaker_verification").lower() in ("true", "yes", "1")
    speaker_path = get_current_config_value(
        "profile_path", "~/.local/state/adam/speaker-profile.npz", section="speaker_verification"
    )
    profile_path = Path(speaker_path).expanduser()
    if not speaker_enabled or not profile_path.is_file():
        print(f"{CLR_YELLOW}Wake-free actions need an enrolled speaker profile to avoid reacting to other voices or media.{CLR_RESET}")
        needs_enrollment = not profile_path.is_file()
        prompt = "Enable speaker verification and enroll your voice now?" if needs_enrollment else "Enable speaker verification with the existing voice profile?"
        if not prompt_yes_no(prompt, default_yes=True):
            update_config_value("enabled", "false", section="idea_routing")
            return
        update_config_value("enabled", "true", section="speaker_verification")
        if needs_enrollment:
            try:
                from src.stt.enroll import main as run_enroll
                run_enroll()
            except Exception as exc:
                print(f"{CLR_RED}Voice enrollment failed: {exc}{CLR_RESET}")
        if not profile_path.is_file():
            print(f"{CLR_YELLOW}No profile was created. Idea routing remains disabled.{CLR_RESET}")
            update_config_value("enabled", "false", section="idea_routing")
            return

    if os.environ.get("ADAM_SKIP_PYTHON") == "1":
        import importlib.util
        if not all(importlib.util.find_spec(module) for module in ("onnxruntime", "tokenizers", "huggingface_hub")):
            print(f"{CLR_YELLOW}Embedding dependencies are missing and --skip-python-deps prevents installing them.{CLR_RESET}")
            update_config_value("enabled", "false", section="idea_routing")
            return
    else:
        uv = shutil.which("uv")
        if not uv:
            print(f"{CLR_RED}uv was not found; idea routing dependencies could not be installed.{CLR_RESET}")
            update_config_value("enabled", "false", section="idea_routing")
            return
        sync_args = [uv, "sync", "--extra", os.environ.get("ADAM_RUNTIME_EXTRA", "runtime-cpu"), "--extra", "intent-routing"]
        if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") != "1":
            sync_args.extend(["--extra", "speaker-verification"])
        if get_current_config_value("enabled", "false", section="speaker_diarization").lower() in ("true", "yes", "1"):
            sync_args.extend(["--extra", "nemotron-diarization"])
        try:
            subprocess.run(sync_args, cwd=PROJECT_DIR, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"{CLR_YELLOW}Could not install idea-routing dependencies ({exc}); leaving the feature disabled.{CLR_RESET}")
            update_config_value("enabled", "false", section="idea_routing")
            return

    if not prompt_yes_no("Download the quantized MiniLM model now (~23 MB on supported CPUs)?", default_yes=True):
        print(f"{CLR_YELLOW}Model download skipped. Run 'uv run python -m src.intent.idea_router --download' later, then enable idea_routing in config.yaml.{CLR_RESET}")
        update_config_value("enabled", "false", section="idea_routing")
        return
    uv = shutil.which("uv")
    if not uv:
        print(f"{CLR_RED}uv was not found; download the MiniLM model with the documented command after installing uv.{CLR_RESET}")
        update_config_value("enabled", "false", section="idea_routing")
        return
    try:
        subprocess.run(
            [uv, "run", "--no-sync", "python", "-m", "src.intent.idea_router", "--download"],
            cwd=PROJECT_DIR,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{CLR_YELLOW}MiniLM setup failed ({exc}); leaving the feature disabled until download succeeds.{CLR_RESET}")
        update_config_value("enabled", "false", section="idea_routing")
        return

    update_config_value("enabled", "true", section="idea_routing")
    print(f"{CLR_GREEN}Idea routing enabled. Edit assets/intent_ideas.json to change the ideas Adam matches.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 11: Existing Browser
# ------------------------------------------------------------------------------
def configure_browser_navigation():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 11: Browser Navigation ---{CLR_RESET}")
    current_enabled = get_current_config_value(
        "enabled", "false", section="browser_navigation"
    ).lower() in ("true", "yes", "1")
    requested_default = os.environ.get("ADAM_BROWSER_NAVIGATION_DEFAULT") == "1"
    default_enabled = current_enabled or requested_default
    if not prompt_yes_no(
        "Enable safe browser navigation in Adam's separate browser profile?",
        default_yes=default_enabled,
    ):
        update_config_value("enabled", "false", section="browser_navigation")
        print(f"{CLR_YELLOW}Separate-profile browser navigation remains disabled.{CLR_RESET}")
        return

    if not ensure_browser_navigation_ready():
        update_config_value("enabled", "false", section="browser_navigation")
        return

    # Do not change profile_path. Its default is a dedicated Adam profile,
    # separate from the user's regular browser data.
    update_config_value("enabled", "true", section="browser_navigation")
    profile_path = Path(
        get_current_config_value(
            "profile_path", "~/.local/share/adam/browser-navigation", section="browser_navigation"
        )
    ).expanduser()
    print(
        f"{CLR_GREEN}Browser navigation is ready. Adam will use its separate profile at "
        f"{profile_path}.{CLR_RESET}"
    )


def browser_navigation_target() -> tuple[str, bool, str]:
    """Return (Playwright engine, needs managed browser, readiness detail)."""
    requested = get_current_config_value("browser", "default", section="browser_navigation").lower()
    configured_default = get_current_config_value(
        "default_browser", "chromium", section="desktop"
    ).lower()
    browser_name = configured_default if requested == "default" else requested

    if "firefox" in browser_name or "mozilla" in browser_name:
        return "firefox", True, "Playwright's Firefox build"
    if not any(token in browser_name for token in ("chrom", "edge", "msedge", "chrome", "brave", "vivaldi", "zen", "opera")):
        return "", False, f"unsupported browser {browser_name!r}; choose Chromium or Firefox"

    if requested == "chromium" or browser_name in ("chromium", "chromium-browser"):
        return "chromium", True, "Playwright's Chromium build"

    candidates: tuple[str, ...]
    if "edge" in browser_name or "msedge" in browser_name:
        candidates = (browser_name, "microsoft-edge", "msedge")
        if any(shutil.which(candidate) for candidate in candidates):
            return "chromium", False, "configured Edge browser executable"
        return "", False, "the configured Edge browser is not installed; install it or choose browser_navigation.browser: chromium"
    elif "google-chrome" in browser_name or browser_name == "chrome":
        candidates = (browser_name, "google-chrome", "google-chrome-stable", "chrome")
        if any(shutil.which(candidate) for candidate in candidates):
            return "chromium", False, "configured Chrome browser executable"
        return "", False, "the configured Chrome browser is not installed; install it or choose browser_navigation.browser: chromium"
    elif "brave" in browser_name:
        candidates = (browser_name, "brave-browser", "brave")
    elif "vivaldi" in browser_name:
        candidates = (browser_name, "vivaldi-stable", "vivaldi")
    elif "zen" in browser_name:
        candidates = (browser_name, "zen", "zen-browser")
    elif "opera" in browser_name:
        candidates = (browser_name, "opera")
    else:
        candidates = (browser_name,)

    if any(shutil.which(candidate) for candidate in candidates):
        return "chromium", False, "configured Chromium browser executable"
    # BrowserNavigator falls back to Playwright's bundled Chromium for these
    # Chromium-family names when no branded executable is found.
    return "chromium", True, "Playwright's Chromium build"


def ensure_browser_navigation_ready() -> bool:
    """Prepare browser-control only after the user opts into the feature."""
    engine, needs_managed_browser, detail = browser_navigation_target()
    if not engine:
        print(f"{CLR_YELLOW}Browser navigation was not enabled: {detail}.{CLR_RESET}")
        print("Set browser_navigation.browser to 'chromium' or 'firefox' and rerun setup.")
        return False

    if importlib.util.find_spec("playwright") is None:
        if os.environ.get("ADAM_SKIP_PYTHON") == "1":
            print(f"{CLR_YELLOW}Playwright is not installed and --skip-python-deps prevents installing it.{CLR_RESET}")
            print("After setup, run: uv pip install --python .venv/bin/python 'playwright>=1.50.0'")
            print("Then rerun ./setup.sh --browser-navigation to finish browser setup.")
            return False
        uv = shutil.which("uv")
        if not uv:
            print(f"{CLR_YELLOW}Playwright is missing and uv is unavailable; browser navigation remains disabled.{CLR_RESET}")
            print("Install Playwright into .venv, then rerun ./setup.sh --browser-navigation.")
            return False
        try:
            subprocess.run(
                [uv, "pip", "install", "--python", str(PROJECT_DIR / ".venv" / "bin" / "python"), "playwright>=1.50.0"],
                cwd=PROJECT_DIR,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"{CLR_YELLOW}Could not install Playwright ({exc}); browser navigation remains disabled.{CLR_RESET}")
            print("After installing Playwright, rerun ./setup.sh --browser-navigation.")
            return False
        if importlib.util.find_spec("playwright") is None:
            print(f"{CLR_YELLOW}Playwright still cannot be imported; browser navigation remains disabled.{CLR_RESET}")
            return False

    if needs_managed_browser:
        if os.environ.get("ADAM_SKIP_PYTHON") == "1":
            print(f"{CLR_YELLOW}The {detail} is not guaranteed to be installed and --skip-python-deps was selected.{CLR_RESET}")
            print(f"After setup, run: uv run --no-sync playwright install {engine}")
            print("Then rerun ./setup.sh --browser-navigation to enable the feature.")
            return False
        uv = shutil.which("uv")
        if not uv:
            print(f"{CLR_YELLOW}uv is unavailable, so {detail} could not be prepared.{CLR_RESET}")
            print(f"Install it with 'playwright install {engine}', then rerun setup.")
            return False
        try:
            subprocess.run(
                [uv, "run", "--no-sync", "playwright", "install", engine],
                cwd=PROJECT_DIR,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"{CLR_YELLOW}Could not prepare {detail} ({exc}); browser navigation remains disabled.{CLR_RESET}")
            print(f"After installing it with 'uv run --no-sync playwright install {engine}', rerun setup.")
            return False
    return True


# ------------------------------------------------------------------------------
# Step 12: OCR-only Desktop Use and Jev
# ------------------------------------------------------------------------------
def configure_ocr_computer_use():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 12: OCR Desktop Use with Jev ---{CLR_RESET}")
    print(
        "Adam can keep screenshot pixels out of the model context and use OCR text boxes instead. "
        "TypeSafe Jev can choose the next desktop operation and recognized text target across native apps and browsers."
    )
    section = "computer_control"
    current = get_current_config_value("ocr_only", "false", section=section).lower() in ("true", "yes", "1")
    if not prompt_yes_no("Use OCR-only screen observations (never attach screenshot pixels)?", default_yes=current):
        update_config_value("ocr_only", "false", section=section)
        update_config_value("jev_enabled", "false", section=section)
        return

    if os.environ.get("ADAM_SKIP_PYTHON") == "1":
        import importlib.util
        if not importlib.util.find_spec("rapidocr"):
            print(f"{CLR_YELLOW}OCR package is missing and --skip-python-deps prevents installing it; OCR-only mode remains disabled.{CLR_RESET}")
            update_config_value("ocr_only", "false", section=section)
            update_config_value("jev_enabled", "false", section=section)
            return
    else:
        uv = shutil.which("uv")
        if not uv:
            print(f"{CLR_RED}uv was not found; OCR dependencies could not be installed.{CLR_RESET}")
            update_config_value("ocr_only", "false", section=section)
            update_config_value("jev_enabled", "false", section=section)
            return
        sync_args = [uv, "sync", "--extra", os.environ.get("ADAM_RUNTIME_EXTRA", "runtime-cpu"), "--extra", "computer-ocr"]
        if os.environ.get("ADAM_SKIP_SPEAKER_VERIFICATION") != "1":
            sync_args.extend(["--extra", "speaker-verification"])
        if get_current_config_value("enabled", "false", section="idea_routing").lower() in ("true", "yes", "1"):
            sync_args.extend(["--extra", "intent-routing"])
        if get_current_config_value("enabled", "false", section="speaker_diarization").lower() in ("true", "yes", "1"):
            sync_args.extend(["--extra", "nemotron-diarization"])
        if get_current_config_value("enabled", "false", section="browser_navigation").lower() in ("true", "yes", "1"):
            sync_args.extend(["--extra", "browser-control"])
        try:
            subprocess.run(sync_args, cwd=PROJECT_DIR, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"{CLR_YELLOW}Could not install OCR dependencies ({exc}); OCR-only mode remains disabled.{CLR_RESET}")
            update_config_value("ocr_only", "false", section=section)
            update_config_value("jev_enabled", "false", section=section)
            return

    print(f"{CLR_CYAN}Preparing PP-OCRv6 medium weights (about 133 MB; downloaded once)...{CLR_RESET}")
    preload = [
        sys.executable, "-c",
        "from src.tools.ocr import ScreenOCR; ScreenOCR()._get_engine()",
    ] if os.environ.get("ADAM_SKIP_PYTHON") == "1" else [
        uv, "run", "--no-sync", "python", "-c",
        "from src.tools.ocr import ScreenOCR; ScreenOCR()._get_engine()",
    ]
    try:
        subprocess.run(preload, cwd=PROJECT_DIR, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{CLR_YELLOW}Could not prepare PP-OCRv6 medium ({exc}); OCR-only mode remains disabled.{CLR_RESET}")
        update_config_value("ocr_only", "false", section=section)
        update_config_value("jev_enabled", "false", section=section)
        return

    update_config_value("ocr_only", "true", section=section)
    jev_current = get_current_config_value("jev_enabled", "false", section=section).lower() in ("true", "yes", "1")
    if prompt_yes_no("Use TypeSafe Jev 1.13 for desktop action and target decisions?", default_yes=jev_current):
        api_key = get_current_config_value("api_key", "", section="llm") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            import getpass
            api_key = getpass.getpass("OpenRouter API key for Jev (input hidden): ").strip()
            if api_key:
                update_config_value("api_key", api_key, section="llm")
        if not api_key:
            update_config_value("jev_enabled", "false", section=section)
            print(f"{CLR_YELLOW}Jev remains disabled until an OpenRouter API key is configured in llm.api_key or OPENROUTER_API_KEY.{CLR_RESET}")
            return
        update_config_value("jev_enabled", "true", section=section)
        update_config_value("jev_api_base", "https://openrouter.ai/api/alpha/decisions", section=section)
        update_config_value("jev_model", "typesafe/jev-1.13", section=section)
        print(f"{CLR_GREEN}Jev 1.13 desktop decisions configured with your OpenRouter key. No local model service is required.{CLR_RESET}")
    else:
        update_config_value("jev_enabled", "false", section=section)
        print(f"{CLR_GREEN}OCR-only desktop observations enabled without Jev action selection.{CLR_RESET}")


# Step 13: Optional Screenshot Grounding
# ------------------------------------------------------------------------------
def configure_computer_vision():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 13: Optional Screenshot Grounding (OmniParser) ---{CLR_RESET}")
    if get_current_config_value("ocr_only", "false", section="computer_control").lower() in ("true", "yes", "1"):
        update_config_value("enabled", "false", section="computer_vision")
        print(f"{CLR_YELLOW}OCR-only mode is active, so screenshot grounding is disabled.{CLR_RESET}")
        return
    print(
        "OmniParser's YOLOv8 Nano detector adds numbered clickable-region boxes and screenshot-pixel coordinates. "
        "It does not caption controls; Adam reads their visible text from the screenshot. "
        "The isolated runtime downloads about 41 MB of weights. Vulkan is not supported by this detector. "
        "The YOLOv8 checkpoint and Ultralytics runtime are AGPL-3.0 licensed."
    )
    section = "computer_vision"
    enabled_now = get_current_config_value("enabled", "false", section=section).lower() in ("true", "yes", "1")
    if not prompt_yes_no("Enable OmniParser screenshot boxes?", default_yes=enabled_now):
        update_config_value("enabled", "false", section=section)
        return

    gpu_rows = []
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for row in result.stdout.splitlines():
                fields = [field.strip() for field in row.split(",", 2)]
                if len(fields) == 3:
                    gpu_rows.append(fields)

    current_device = get_current_config_value("device", "cuda", section=section).lower()
    devices = ["CPU (portable; slower)"]
    if gpu_rows:
        devices.extend([f"CUDA: GPU {index} — {name}" for index, name, _ in gpu_rows])
    default_device = 0
    if current_device == "cuda" and gpu_rows:
        current_uuid = get_current_config_value("gpu_uuid", "", section=section)
        default_device = next(
            (i + 1 for i, (_, _, uuid) in enumerate(gpu_rows) if uuid == current_uuid), 1
        )
    choice = prompt_choice("Select OmniParser inference device", devices, default_idx=default_device)
    if choice == 0:
        device, gpu_uuid = "cpu", ""
    else:
        device, gpu_uuid = "cuda", gpu_rows[choice - 1][2]

    installer = PROJECT_DIR / "tools" / "install_omniparser.py"
    command = [sys.executable, str(installer), "--device", device]
    if gpu_uuid:
        command.extend(["--gpu-uuid", gpu_uuid])
    print(f"{CLR_CYAN}Installing the isolated {device.upper()} runtime and OmniParser detector...{CLR_RESET}")
    installed = subprocess.run(command, cwd=PROJECT_DIR).returncode == 0
    if not installed:
        update_config_value("enabled", "false", section=section)
        print(f"{CLR_YELLOW}OmniParser setup did not complete; screenshot grounding remains disabled.{CLR_RESET}")
        return

    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")).expanduser()
    update_config_value("backend", "omniparser", section=section)
    update_config_value("device", device, section=section)
    update_config_value("gpu_uuid", gpu_uuid, section=section)
    update_config_value("python_path", str(data_home / "adam" / "omniparser-runtime" / "bin" / "python"), section=section)
    update_config_value("model_path", str(data_home / "adam" / "models" / "omniparser-yolov8n.pt"), section=section)
    update_config_value("enabled", "true", section=section)
    print(f"{CLR_GREEN}OmniParser boxes enabled on {device.upper()}.{CLR_RESET}")


# Step 14: Systemd Daemon Setup
# ------------------------------------------------------------------------------
def ensure_onnxruntime() -> bool:
    """Repair a missing or damaged ONNX Runtime install before starting Adam."""
    runtime_extra = os.environ.get("ADAM_RUNTIME_EXTRA", "runtime-cpu")
    package = "onnxruntime-gpu" if runtime_extra == "runtime-nvidia" else "onnxruntime"
    python = PROJECT_DIR / ".venv" / "bin" / "python"
    check = [str(python), "-c", "import onnxruntime"]
    if subprocess.run(check, cwd=PROJECT_DIR, capture_output=True).returncode == 0:
        return True

    if os.environ.get("ADAM_SKIP_PYTHON") == "1":
        print(f"{CLR_RED}ONNX Runtime cannot be imported and Python dependency installation was skipped; Adam will not be started.{CLR_RESET}")
        return False
    uv = shutil.which("uv")
    if not uv:
        print(f"{CLR_RED}ONNX Runtime cannot be imported and uv is unavailable; Adam will not be started.{CLR_RESET}")
        return False

    print(f"{CLR_YELLOW}ONNX Runtime files are missing or damaged; reinstalling {package} before starting Adam.{CLR_RESET}")
    try:
        subprocess.run(
            [uv, "pip", "install", "--reinstall", "--python", str(python), package],
            cwd=PROJECT_DIR,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"{CLR_RED}Could not repair {package} ({exc}); Adam will not be started.{CLR_RESET}")
        return False
    if subprocess.run(check, cwd=PROJECT_DIR, capture_output=True).returncode != 0:
        print(f"{CLR_RED}{package} is still not importable after reinstall; Adam will not be started.{CLR_RESET}")
        return False
    print(f"{CLR_GREEN}{package} is importable.{CLR_RESET}")
    return True


def ensure_torchaudio() -> bool:
    """Do not start the service with mismatched/unloadable PyTorch audio wheels."""
    python = PROJECT_DIR / ".venv" / "bin" / "python"
    result = subprocess.run(
        [str(python), "-c", "import torch, torchaudio"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    detail = (result.stderr or result.stdout or "import failed").strip().splitlines()[-1]
    print(f"{CLR_RED}PyTorch/TorchAudio cannot be imported together ({detail}); Adam will not be started. Rerun ./setup.sh to repair the runtime.{CLR_RESET}")
    return False


def systemd_user_manager_available() -> bool:
    if not shutil.which("systemctl") or not Path("/run/systemd/system").exists():
        return False
    try:
        return subprocess.run(
            ["systemctl", "--user", "show-environment"], capture_output=True
        ).returncode == 0
    except OSError:
        return False


def _run_user_systemctl(*args: str):
    command = ["systemctl", "--user", *args]
    try:
        return subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _report_systemctl_failure(action: str, result) -> None:
    detail = (result.stderr or result.stdout or "no error details").strip()
    print(f"{CLR_RED}Could not {action} ({detail}).{CLR_RESET}")


def _unit_path_value_is_safe(value: str) -> bool:
    # Keep values unambiguous in systemd's assignment and ExecStart syntaxes.
    return bool(re.fullmatch(r"[A-Za-z0-9_./+-]+", value))


def render_systemd_unit() -> str | None:
    template_path = PROJECT_DIR / "systemd" / "adam.service.template"
    if not template_path.is_file():
        print(f"{CLR_YELLOW}Systemd template is missing: {template_path}{CLR_RESET}")
        return None

    project_dir = str(PROJECT_DIR.resolve())
    home_dir = str(Path.home().resolve())
    if not _unit_path_value_is_safe(project_dir) or not _unit_path_value_is_safe(home_dir):
        print(
            f"{CLR_YELLOW}The project or home path contains characters this service template "
            f"cannot safely represent; leaving the unit unchanged.{CLR_RESET}"
        )
        return None
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"{CLR_YELLOW}Could not read systemd template ({exc}); leaving the unit unchanged.{CLR_RESET}")
        return None
    rendered = template.replace("{{PROJECT_DIR}}", project_dir).replace("{{HOME}}", home_dir)
    if "{{PROJECT_DIR}}" in rendered or "{{HOME}}" in rendered:
        print(f"{CLR_YELLOW}Systemd template has unresolved path placeholders; leaving the unit unchanged.{CLR_RESET}")
        return None
    return rendered


def _atomic_write_text(destination: Path, content: str, mode: int = 0o644) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def backup_existing_unit(unit_path: Path) -> Path | None:
    index = 0
    while True:
        suffix = ".backup" if index == 0 else f".backup.{index}"
        backup_path = unit_path.with_name(unit_path.name + suffix)
        if not backup_path.exists():
            break
        index += 1
    fd, temporary_name = tempfile.mkstemp(prefix=f".{backup_path.name}.", dir=backup_path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(unit_path, temporary_path)
        os.replace(temporary_path, backup_path)
    except OSError as exc:
        print(f"{CLR_RED}Could not back up the existing unit ({exc}); it was not replaced.{CLR_RESET}")
        return None
    finally:
        temporary_path.unlink(missing_ok=True)
    return backup_path


def _configured_tts_asset_paths() -> list[Path]:
    paths = []
    for key, default in (
        ("model_path", "assets/voices/kokoro/kokoro-v1.0.onnx"),
        ("voices_path", "assets/voices/kokoro/voices-v1.0.bin"),
    ):
        value = Path(get_current_config_value(key, default, section="tts")).expanduser()
        paths.append(value if value.is_absolute() else PROJECT_DIR / value)
    return paths


def configure_systemd():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 14: Systemd User Service ---{CLR_RESET}")
    if os.environ.get("ADAM_SKIP_SERVICE") == "1":
        print(f"{CLR_YELLOW}Service setup skipped by setup option.{CLR_RESET}")
        return
    if not systemd_user_manager_available():
        print(f"{CLR_YELLOW}This Linux system does not use systemd; start Adam with 'uv run python -m src.main' or use your init system.{CLR_RESET}")
        return

    if not prompt_yes_no(
        "Install and enable Adam as a background user service (adam.service)?",
        default_yes=False,
    ):
        print(f"{CLR_YELLOW}Service setup skipped. You can start Adam manually with 'uv run python -m src.main'.{CLR_RESET}")
        return

    if get_current_config_value("engine", "kokoro", section="tts").lower() == "kokoro":
        missing_assets = [path for path in _configured_tts_asset_paths() if not path.is_file()]
        if missing_assets:
            print(f"{CLR_YELLOW}Adam was not enabled because Kokoro assets are missing:{CLR_RESET}")
            for asset in missing_assets:
                print(f"  - {asset}")
            print("Rerun ./setup.sh and allow the model downloads, then enable the service.")
            return

    if not ensure_onnxruntime() or not ensure_torchaudio():
        return

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_path = service_dir / "adam.service"
    if service_path.is_symlink() or (service_path.exists() and not service_path.is_file()):
        print(f"{CLR_YELLOW}{service_path} is not a regular unit file; leaving it unchanged.{CLR_RESET}")
        return

    unit_exists = service_path.is_file()
    should_write = not unit_exists
    rendered = None
    if unit_exists:
        if prompt_yes_no(
            "Replace the existing adam.service? A backup will be kept beside it.",
            default_yes=False,
        ):
            rendered = render_systemd_unit()
            if rendered is None:
                return
            should_write = True
        else:
            print(f"{CLR_CYAN}Keeping the existing adam.service unchanged and using it.{CLR_RESET}")
    else:
        rendered = render_systemd_unit()
        if rendered is None:
            return

    # Do not disrupt a legacy service until the replacement has passed every
    # startup preflight, including validation of the unit path and template.
    legacy_active = _run_user_systemctl("is-active", "adam-kev.service")
    legacy_enabled = _run_user_systemctl("is-enabled", "adam-kev.service")
    if legacy_active.returncode == 0 or legacy_enabled.returncode == 0:
        print(f"{CLR_CYAN}A retired adam-kev.service is active or enabled.{CLR_RESET}")
        if prompt_yes_no("Stop and disable adam-kev.service?", default_yes=False):
            result = _run_user_systemctl("disable", "--now", "adam-kev.service")
            if result.returncode != 0:
                _report_systemctl_failure("stop and disable adam-kev.service", result)
                return
            print(f"{CLR_GREEN}adam-kev.service was stopped and disabled.{CLR_RESET}")
        else:
            print(f"{CLR_YELLOW}Leaving adam-kev.service unchanged as requested.{CLR_RESET}")

    if should_write:
        if unit_exists:
            backup_path = backup_existing_unit(service_path)
            if backup_path is None:
                return
            print(f"{CLR_CYAN}Backed up the previous unit to {backup_path}.{CLR_RESET}")
        try:
            _atomic_write_text(service_path, rendered)
        except OSError as exc:
            if unit_exists:
                print(f"{CLR_RED}Could not replace adam.service ({exc}). The backup is available at {backup_path}.{CLR_RESET}")
            else:
                print(f"{CLR_RED}Could not install adam.service ({exc}).{CLR_RESET}")
            return
        if not unit_exists:
            print(f"{CLR_GREEN}Installed {service_path}.{CLR_RESET}")

    if should_write:
        result = _run_user_systemctl("daemon-reload")
        if result.returncode != 0:
            _report_systemctl_failure("reload the user service manager", result)
            return

    print(f"{CLR_CYAN}Enabling and starting adam.service...{CLR_RESET}")
    result = _run_user_systemctl("enable", "--now", "adam.service")
    if result.returncode != 0:
        _report_systemctl_failure("enable and start adam.service", result)
        return
    status = _run_user_systemctl("is-active", "adam.service")
    if status.returncode == 0:
        print(f"{CLR_GREEN}adam.service is active.{CLR_RESET}")
    else:
        print(f"{CLR_YELLOW}adam.service was enabled, but systemd did not report it active. Check: journalctl --user -u adam.service -n 50{CLR_RESET}")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    print_banner()
    try:
        configure_audio()
        configure_persona()
        configure_stt()
        configure_tts()
        configure_llm()
        configure_browser()
        configure_speaker_verification()
        configure_meeting()
        configure_speaker_diarization()
        configure_idea_routing()
        configure_browser_navigation()
        configure_ocr_computer_use()
        configure_computer_vision()
        config_path = PROJECT_DIR / "config.yaml"
        if config_path.exists():
            config_path.chmod(0o600)
        configure_systemd()

        print(f"\n{CLR_GREEN}{CLR_BOLD}================================================================{CLR_RESET}")
        print(f"{CLR_GREEN}{CLR_BOLD}          Setup Complete! Adam / Adam is Ready.                {CLR_RESET}")
        print(f"{CLR_GREEN}{CLR_BOLD}================================================================{CLR_RESET}")
        print(f"  • View live logs:    {CLR_BOLD}journalctl --user -u adam.service -f{CLR_RESET}")
        print(f"  • Re-enroll voice:   {CLR_BOLD}uv run python -m src.stt.enroll{CLR_RESET}")
        print(f"  • Test microphone:   {CLR_BOLD}uv run python tools/test_mic.py{CLR_RESET}")
        print(f"  • Meeting recordings: {CLR_BOLD}{Path(get_current_config_value('output_dir', '~/.local/state/adam/meetings', section='meeting')).expanduser()}{CLR_RESET}")
        print(f"  • Heard audio/logs:  {CLR_BOLD}{Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'adam' / 'heard-captures'}{CLR_RESET}")
        print(f"  • Idea definitions: {CLR_BOLD}{PROJECT_DIR / 'assets' / 'intent_ideas.json'}{CLR_RESET}")
        print(f"  • Edit config:       {CLR_BOLD}{PROJECT_DIR}/config.yaml{CLR_RESET}\n")
    except KeyboardInterrupt:
        print(f"\n\n{CLR_YELLOW}Setup cancelled by user.{CLR_RESET}\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
