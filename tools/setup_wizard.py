#!/usr/bin/env python3
"""Comprehensive interactive setup wizard for Shin / Adam Voice Assistant.

Configures:
- Audio hardware (Microphone inputs, playback outputs)
- Assistant persona & wake word
- Speech-to-Text (STT) engine & hardware acceleration (Vulkan, CUDA, CPU)
- Text-to-Speech (TTS) engine, voice & device
- LLM sourcing (Local Ollama, custom vLLM/llama.cpp server, or Cloud APIs: Groq/Gemini/Claude/OpenAI)
- Default web browser
- Voice profile enrollment (speaker verification)
- Systemd background user service
"""

import os
import re
import sys
import subprocess
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


def print_banner():
    print(f"""
{CLR_CYAN}{CLR_BOLD}================================================================
          🎙️   Shin / Adam Voice Assistant Setup
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


def update_config_value(key: str, value: str):
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

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count > 0:
        config_path.write_text(new_content, encoding="utf-8")
    else:
        # If key didn't exist in config, we don't break the file
        pass


def get_current_config_value(key: str, default: str = "") -> str:
    """Reads a current configuration string value from config.yaml or config.yaml.example."""
    config_path = PROJECT_DIR / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_DIR / "config.yaml.example"
    if not config_path.exists():
        return default
    content = config_path.read_text(encoding="utf-8")
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
def configure_stt():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 3: Speech-to-Text (STT) Engine & Hardware ---{CLR_RESET}")
    current_model = get_current_config_value("model_size", "qwen3-asr-1.7b")
    current_dev = get_current_config_value("device", "Vulkan0")

    stt_options = [
        "Qwen3-ASR-1.7B on GPU / Vulkan (Recommended: fast, high accuracy, low latency)",
        "Faster-Whisper (distil-large-v3) on NVIDIA CUDA (Pascal/Ampere/Ada)",
        "Faster-Whisper (base.en) on CPU (Universal: low resource, no GPU needed)",
        "Custom STT model & device",
    ]

    default_idx = 0
    if "distil" in current_model.lower():
        default_idx = 1
    elif "base" in current_model.lower() or current_dev == "cpu":
        default_idx = 2

    choice = prompt_choice("Select STT engine configuration", stt_options, default_idx)

    if choice == 0:
        update_config_value("model_size", "qwen3-asr-1.7b")
        update_config_value("device", "Vulkan0")
        update_config_value("compute_type", "int8_float32")
        print(f"{CLR_GREEN}Configured Qwen3-ASR on Vulkan0.{CLR_RESET}")
    elif choice == 1:
        update_config_value("model_size", "distil-large-v3")
        update_config_value("device", "cuda")
        update_config_value("compute_type", "int8_float32")
        print(f"{CLR_GREEN}Configured faster-whisper distil-large-v3 on CUDA.{CLR_RESET}")
    elif choice == 2:
        update_config_value("model_size", "base.en")
        update_config_value("device", "cpu")
        update_config_value("compute_type", "int8")
        print(f"{CLR_GREEN}Configured faster-whisper base.en on CPU.{CLR_RESET}")
    else:
        custom_model = prompt_text("Enter STT model name (e.g. 'qwen3-asr-1.7b', 'small.en', 'medium.en')", current_model)
        custom_dev = prompt_text("Enter compute device ('cuda', 'Vulkan0', 'cpu')", current_dev)
        update_config_value("model_size", custom_model)
        update_config_value("device", custom_dev)
        print(f"{CLR_GREEN}Configured custom STT:{CLR_RESET} model={custom_model}, device={custom_dev}")


# ------------------------------------------------------------------------------
# Step 4: Text-to-Speech (TTS) Engine & Voice
# ------------------------------------------------------------------------------
def configure_tts():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 4: Text-to-Speech (TTS) Engine & Voice ---{CLR_RESET}")
    current_engine = get_current_config_value("engine", "kokoro")
    current_voice = get_current_config_value("voice", "am_adam")

    engines = [
        "Kokoro-82M Neural TTS (High fidelity, natural phrasing)",
        "Piper TTS (Ultra-lightweight fast CPU synthesis)",
    ]
    engine_idx = 0 if current_engine == "kokoro" else 1
    chosen_engine_idx = prompt_choice("Select TTS engine", engines, engine_idx)

    if chosen_engine_idx == 0:
        update_config_value("engine", "kokoro")
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
            update_config_value("voice", voice_map[v_choice])
            print(f"{CLR_GREEN}Configured voice:{CLR_RESET} {voice_map[v_choice]}")
        else:
            custom_v = prompt_text("Enter voice code (e.g. 'af_sarah', 'am_fenrir')", current_voice)
            update_config_value("voice", custom_v)
            print(f"{CLR_GREEN}Configured custom voice:{CLR_RESET} {custom_v}")
    else:
        update_config_value("engine", "piper")
        print(f"{CLR_GREEN}Configured Piper TTS engine.{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 5: LLM Sourcing & Models (Local vs Cloud, Endpoints, Models)
# ------------------------------------------------------------------------------
def configure_llm():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 5: LLM Brain Sourcing & Model Configuration ---{CLR_RESET}")
    current_provider = get_current_config_value("provider", "local")

    source_options = [
        "Local LLM (Run locally on your machine / home server via Ollama or OpenAI-compatible server)",
        "Cloud LLM API (Ultra-fast cloud inference: Groq, Gemini, Anthropic Claude, OpenAI)",
    ]
    default_source = 0 if current_provider in ("local", "custom", "openai_compatible", "vllm") else 1
    source_choice = prompt_choice("Where should Shin source the LLM brain?", source_options, default_source)

    if source_choice == 0:
        # Local Setup
        local_backends = [
            "Ollama (Default: http://localhost:11434)",
            "Custom / Remote OpenAI-compatible endpoint (vLLM, llama.cpp server, LM Studio, LocalAI, or remote Ollama)",
        ]
        curr_endpoint = get_current_config_value("ollama_host", "http://localhost:11434")
        default_backend = 0 if "11434" in curr_endpoint and current_provider == "local" else 1

        b_choice = prompt_choice("Select local runtime backend", local_backends, default_backend)

        if b_choice == 0:
            update_config_value("provider", "local")
            update_config_value("ollama_host", "http://localhost:11434")
            endpoint = "http://localhost:11434"
        else:
            update_config_value("provider", "openai_compatible")
            endpoint = prompt_text("Enter local/remote server endpoint URL", curr_endpoint if curr_endpoint else "http://localhost:8000/v1")
            update_config_value("ollama_host", endpoint)
            update_config_value("api_base", endpoint)
            api_key = prompt_text("Optional API key / Bearer token (leave blank if none)", "")
            if api_key:
                update_config_value("api_key", api_key)

        # Select model
        curr_model = get_current_config_value("local_model", "qwen3.5:4b")
        models = [
            "qwen3.5:4b   - (Recommended) Fast 4B parameter model with 16k context (~2.5GB VRAM)",
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

        update_config_value("local_model", target_model)
        print(f"{CLR_GREEN}Local model configured:{CLR_RESET} {target_model} at {endpoint}")

        # If standard local Ollama is available, offer to pull
        if b_choice == 0 and subprocess.run(["command", "-v", "ollama"], shell=True, stdout=subprocess.DEVNULL).returncode == 0:
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
        ]
        p_choice = prompt_choice("Select cloud provider", cloud_providers, 0)
        p_keys = {0: "groq", 1: "gemini", 2: "anthropic", 3: "openai"}
        provider_name = p_keys[p_choice]
        update_config_value("provider", provider_name)

        model_defaults = {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini"
        }
        chosen_cloud_model = prompt_text(f"Enter {provider_name.capitalize()} model name", model_defaults[provider_name])
        update_config_value("cloud_model", chosen_cloud_model)

        key_env_var = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }[provider_name]

        existing_key = os.environ.get(key_env_var, get_current_config_value("api_key", ""))
        api_key = prompt_text(f"Enter {key_env_var}", existing_key)
        if api_key:
            update_config_value("api_key", api_key)
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
        "Custom browser command",
    ]

    default_idx = 0
    if "firefox" in current_browser:
        default_idx = 1
    elif "chrome" in current_browser:
        default_idx = 2
    elif "brave" in current_browser:
        default_idx = 3
    elif "zen" in current_browser:
        default_idx = 4
    elif "edge" not in current_browser:
        default_idx = 5

    choice = prompt_choice("Choose default browser for web searches and links", browsers, default_idx)
    browser_map = {
        0: "microsoft-edge-stable",
        1: "firefox",
        2: "google-chrome-stable",
        3: "brave",
        4: "zen-browser"
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
    profile_path = Path.home() / ".local" / "state" / "shin" / "speaker-profile.npz"

    if profile_path.exists():
        print(f"{CLR_GREEN}Existing voice profile detected at:{CLR_RESET} {profile_path}")
        enroll_prompt = "Would you like to re-enroll and replace your voice profile now?"
        default_enroll = False
    else:
        print("Speaker verification ensures the assistant only listens to your unique voice.")
        enroll_prompt = "Would you like to enroll your voice profile now (takes ~30s)?"
        default_enroll = True

    if prompt_yes_no(enroll_prompt, default_yes=default_enroll):
        print(f"\n{CLR_CYAN}Starting enrollment session... Speak clearly when prompted.{CLR_RESET}")
        try:
            from src.stt.enroll import main as run_enroll
            run_enroll()
        except Exception as e:
            print(f"{CLR_RED}Enrollment encountered an error: {e}{CLR_RESET}")
            print(f"You can enroll anytime later by running: {CLR_BOLD}uv run python -m src.stt.enroll{CLR_RESET}")


# ------------------------------------------------------------------------------
# Step 8: Systemd Daemon Setup
# ------------------------------------------------------------------------------
def configure_systemd():
    print(f"\n{CLR_BLUE}{CLR_BOLD}--- Step 8: Systemd User Service ---{CLR_RESET}")
    if prompt_yes_no("Enable and start Shin as an automatic background service (shin.service)?", default_yes=True):
        print(f"{CLR_CYAN}Enabling and starting shin.service...{CLR_RESET}")
        subprocess.run(["systemctl", "--user", "daemon-reload"])
        subprocess.run(["systemctl", "--user", "enable", "--now", "shin.service"])
        print(f"{CLR_GREEN}shin.service is active!{CLR_RESET}")
    else:
        print(f"{CLR_YELLOW}You can start the daemon manually anytime with:{CLR_RESET} systemctl --user start shin.service")


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
        configure_systemd()

        print(f"\n{CLR_GREEN}{CLR_BOLD}================================================================{CLR_RESET}")
        print(f"{CLR_GREEN}{CLR_BOLD}          Setup Complete! Shin / Adam is Ready.                {CLR_RESET}")
        print(f"{CLR_GREEN}{CLR_BOLD}================================================================{CLR_RESET}")
        print(f"  • View live logs:    {CLR_BOLD}journalctl --user -u shin.service -f{CLR_RESET}")
        print(f"  • Re-enroll voice:   {CLR_BOLD}uv run python -m src.stt.enroll{CLR_RESET}")
        print(f"  • Test microphone:   {CLR_BOLD}uv run python tools/test_mic.py{CLR_RESET}")
        print(f"  • Edit config:       {CLR_BOLD}{PROJECT_DIR}/config.yaml{CLR_RESET}\n")
    except KeyboardInterrupt:
        print(f"\n\n{CLR_YELLOW}Setup cancelled by user.{CLR_RESET}\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
