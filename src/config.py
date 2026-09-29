import os
import yaml
from pathlib import Path
from typing import Union, Literal, Any
from pydantic import BaseModel, Field, field_validator

class AudioConfig(BaseModel):
    target_sink: str = "Adam_Playback_Sink"
    target_source: str = "Adam_Clean_Mic"
    sample_rate: int = 48000
    chunk_size: int = 1280
    vad_threshold_idle: float = 0.25
    vad_threshold_speaking: float = 0.85
    vad_silence_duration: float = 1.25

class WakeConfig(BaseModel):
    wake_word: str = "hey jarvis"
    aliases: list[str] = Field(default_factory=list)
    threshold: float = 0.50
    followup_window_seconds: float = 7.0

class STTConfig(BaseModel):
    provider: str = "local"
    model_size: str = "qwen3-asr-1.7b"
    language: str = "English"
    device: str = "Vulkan0"
    device_index: int = 0
    compute_type: str = "int8_float32"
    cloud_model: str = "gpt-transcribe"
    cloud_url: str = "https://api.openai.com/v1/audio/transcriptions"
    api_key: str = ""
    fallback_model: str = "base.en"
    fallback_device: str = "cpu"
    fallback_compute_type: str = "int8"

class SpeakerVerificationConfig(BaseModel):
    enabled: bool = True
    profile_path: str = ""
    threshold: float = 0.25

class SpeakerDiarizationConfig(BaseModel):
    enabled: bool = False
    executable: str = "nemo-speech"
    model: str = "nvidia/Nemotron-3-Diarization"
    device: str = "auto"
    timeout_seconds: float = 45.0

    @field_validator("model", mode="before")
    @classmethod
    def migrate_legacy_gguf_model(cls, value):
        # Earlier setup versions stored a GGUF that NeMo-Speech.cpp cannot run
        # with Nemotron 3's pre-LN architecture. Use the official HF checkpoint.
        if isinstance(value, str) and value.lower().endswith(".gguf"):
            return "nvidia/Nemotron-3-Diarization"
        return value

    @field_validator("device", mode="before")
    @classmethod
    def migrate_legacy_vulkan_device(cls, value):
        if isinstance(value, str) and value.lower().startswith("vulkan"):
            return "auto"
        return value

class MeetingConfig(BaseModel):
    output_dir: str = "~/.local/state/adam/meetings"
    max_duration_hours: float = 4.0
    silence_duration: float = 1.25
    max_segment_seconds: float = 30.0
    speaker_similarity_threshold: float = 0.55

class TTSConfig(BaseModel):
    engine: str = "kokoro" # "kokoro" | "openai" | "cosyvoice" | "silent"
    silent_restore_engine: str = "kokoro"
    model_path: str = "assets/voices/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "assets/voices/kokoro/voices-v1.0.bin"
    voice: str = "af_nicole"
    device_id: int = 0
    speed: float = 1.05
    piper_bin: str = "piper"
    config_path: str = "assets/voices/en_US-ryan-high.onnx.json"
    sample_rate: int = 24000
    cloud_model: str = "gpt-4o-mini-tts"
    cloud_voice: str = "marin"
    api_key: str = ""
    cosyvoice_api_url: str = "http://localhost:50000"
    cosyvoice_model_dir: str = "pretrained_models/CosyVoice2-0.5B"


class LLMConfig(BaseModel):
    provider: str = "local"
    local_model: str = "qwen3.5:4b"
    cloud_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    api_base: str = ""
    api_key: str = ""
    num_ctx: int = 16384
    temperature: float = 0.1
    think: Union[bool, str] = False

class ExecutionConfig(BaseModel):
    workspace_dir: str = "~/workspace"
    downloads_dir: str = "~/Downloads"
    jobs_log_dir: str = "~/.local/state/adam/jobs"
    max_log_mb: int = 50
    preferred_video_codec: str = "av1"

DEFAULT_APPLICATION_ALIASES = {
    "discord": ["vesktop", "discord", "webcord", "armcord"],
    "browser": ["microsoft edge", "google chrome", "firefox", "chromium", "brave"],
    "edge": ["microsoft edge", "microsoft edge (beta)"],
    "chrome": ["google chrome"],
    "terminal": ["alacritty", "kitty", "foot", "konsole", "wezterm", "xterm"],
    "music": ["spotify"],
    "spotify": ["spotify"],
    "code": ["zed", "neovim", "visual studio code", "code", "codium"],
    "editor": ["zed", "neovim", "gedit", "micro", "vim"],
    "zed": ["zed"],
    "resolve": ["davinci resolve"],
    "davinci": ["davinci resolve"],
    "obs": ["obs studio"],
    "steam": ["steam"],
    "cyberpunk": ["cyberpunk 2077"],
    "cs2": ["counter-strike 2"],
    "counter strike": ["counter-strike 2"],
    "wukong": ["black myth: wukong"],
    "black myth": ["black myth: wukong"],
    "hollow knight": ["hollow knight"],
    "ultrakill": ["ultrakill"],
    "ror2": ["risk of rain 2"],
    "risk of rain": ["risk of rain 2"],
    "siege": ["tom clancy's rainbow six siege"],
    "rainbow six": ["tom clancy's rainbow six siege"],
    "street fighter": ["street fighter v"],
    "sfv": ["street fighter v"],
    "files": ["dolphin", "nautilus", "thunar"],
    "file manager": ["dolphin", "nautilus", "thunar"],
    "calculator": ["kcalc", "gnome-calculator", "calc"],
    "settings": ["kde system settings", "system settings"]
}

DEFAULT_WINDOW_ALIASES = {
    "browser": ["edge", "microsoft-edge", "chrome", "google-chrome", "firefox", "chromium", "brave", "zen", "vivaldi", "opera"],
    "edge": ["microsoft-edge", "microsoftedge", "edge"],
    "chrome": ["google-chrome", "chrome"],
    "discord": ["vesktop", "webcord", "discord", "armcord"],
    "terminal": ["alacritty", "kitty", "foot", "konsole", "wezterm", "ghostty", "xterm"],
    "music": ["spotify", "cider", "rhythmbox", "cmus"],
    "spotify": ["spotify"],
    "code": ["code", "codium", "vscode", "zed", "cursor", "neovim", "sublime"],
    "editor": ["code", "codium", "vscode", "zed", "cursor", "neovim", "sublime", "kate"],
    "zed": ["dev.zed.zed", "zed"],
    "files": ["nautilus", "dolphin", "thunar", "nemo", "pcmanfm"],
    "obs": ["com.obsproject.studio", "obs", "obs64"],
    "resolve": ["resolve", "davinciresolve"],
    "steam": ["steam"],
    "cyberpunk": ["cyberpunk2077.exe", "cyberpunk"],
    "cs2": ["cs2", "counterstrike"]
}

class DesktopConfig(BaseModel):
    default_browser: str = "microsoft-edge-stable"
    application_aliases: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_APPLICATION_ALIASES))
    window_aliases: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_WINDOW_ALIASES))
    macros: dict[str, Any] = Field(default_factory=dict)
    disabled_capabilities: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)

class AppConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    speaker_verification: SpeakerVerificationConfig = Field(default_factory=SpeakerVerificationConfig)
    speaker_diarization: SpeakerDiarizationConfig = Field(default_factory=SpeakerDiarizationConfig)
    meeting: MeetingConfig = Field(default_factory=MeetingConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)

def load_config(config_path: str = "config.yaml") -> AppConfig:
    p = Path(config_path)
    if not p.exists():
        example = p.with_name(f"{p.name}.example")
        if example.exists():
            import shutil
            shutil.copyfile(example, p)
        else:
            return AppConfig()
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)
