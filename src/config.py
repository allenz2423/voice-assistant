import os
import yaml
from pathlib import Path
from typing import Union, Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator

class AudioConfig(BaseModel):
    target_sink: str = "Adam_Playback_Sink"
    target_source: str = "Adam_Clean_Mic"
    sample_rate: int = 48000
    chunk_size: int = 1280
    vad_threshold_idle: float = 0.25
    vad_threshold_speaking: float = 0.85
    wake_energy_floor: float = Field(default=0.0005, ge=0.0, le=0.1)
    vad_silence_duration: float = 1.25
    max_utterance_seconds: float = Field(default=60.0, ge=1.0, le=300.0)

class WakeConfig(BaseModel):
    wake_word: str = "hey jarvis"
    aliases: list[str] = Field(default_factory=list)
    threshold: float = 0.50
    followup_window_seconds: float = 7.0

class STTConfig(BaseModel):
    provider: str = "local"
    model_size: str = "qwen3-asr-1.7b"
    language: str = ""
    device: str = "Vulkan0"
    device_index: int = 0
    compute_type: str = "int8_float32"
    cloud_model: str = "gpt-transcribe"
    cloud_url: str = "https://api.openai.com/v1/audio/transcriptions"
    api_key: str = ""
    fallback_model: str = "base.en"
    fallback_device: str = "cpu"
    fallback_compute_type: str = "int8"

class SpeakerUserConfig(BaseModel):
    name: str
    profile_path: str = ""
    profiles: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, name):
        name = name.strip()
        if not name:
            raise ValueError("Speaker user name cannot be empty.")
        return name


class SpeakerVerificationConfig(BaseModel):
    enabled: bool = True
    profile_path: str = ""
    threshold: float = 0.25
    users: list[SpeakerUserConfig] = Field(default_factory=list)

    @field_validator("users", mode="before")
    @classmethod
    def accept_user_name_shorthand(cls, users):
        if users is None:
            return []
        if not isinstance(users, list):
            raise ValueError("Speaker users must be a list of names or user objects.")
        normalized = []
        for user in users:
            if isinstance(user, str):
                normalized.append({"name": user})
            elif isinstance(user, dict) and isinstance(user.get("profiles"), list):
                item = dict(user)
                item["profiles"] = {str(profile): "" for profile in item["profiles"]}
                normalized.append(item)
            else:
                normalized.append(user)
        return normalized

    @field_validator("users")
    @classmethod
    def validate_users(cls, users):
        names = [user.name.strip().casefold() for user in users]
        if any(not name for name in names):
            raise ValueError("Speaker user names cannot be empty.")
        if len(names) != len(set(names)):
            raise ValueError("Speaker user names must be unique, ignoring case.")
        for user in users:
            profiles = [name.strip().casefold() for name in user.profiles]
            if any(not name for name in profiles) or len(profiles) != len(set(profiles)):
                raise ValueError(f"Profile names for speaker user {user.name!r} must be non-empty and unique.")
            if user.profiles and user.profile_path:
                raise ValueError(f"Use either profile_path or profiles for speaker user {user.name!r}, not both.")
        return users

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

class IdeaRoutingConfig(BaseModel):
    enabled: bool = False
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ideas_path: str = "assets/intent_ideas.json"
    command_threshold: float = 0.30
    background_threshold: float = 0.49
    minimum_margin: float = 0.025
    require_enrolled_speaker: bool = True

class BrowserNavigationConfig(BaseModel):
    enabled: bool = False
    browser: str = "default"  # "default", "chromium", or "firefox"
    profile_path: str = "~/.local/share/adam/browser-navigation"
    timeout_seconds: float = 15.0

class ComputerControlConfig(BaseModel):
    enabled: bool = True
    max_text_length: int = 20000
    screenshot_delay_seconds: float = 0.25
    browser_screenshot_delay_seconds: float = 3.0
    ocr_only: bool = False
    jev_enabled: bool = False
    jev_api_base: str = "https://openrouter.ai/api/alpha/decisions"
    jev_model: str = "typesafe/jev-1.13"
    jev_timeout_seconds: float = 20.0
    jev_min_confidence: float = 0.65
    ocr_max_regions: int = 100
    ocr_max_candidates: int = Field(default=800, ge=1, le=5000)
    ocr_max_image_dimension: int = Field(default=1280, ge=320, le=4096)
    ocr_model_size: Literal["small", "medium"] = "small"
    ocr_preload_on_startup: bool = False
    ocr_device: Literal["cpu", "cuda"] | None = None
    ocr_gpu_uuid: str = ""
    max_sequence_actions: int = Field(default=8, ge=1, le=8)
    max_sequence_text_length: int = Field(default=20000, ge=1, le=160000)
    sequence_timeout_seconds: float = Field(default=45.0, ge=1.0, le=120.0)

class ComputerVisionConfig(BaseModel):
    enabled: bool = False
    backend: Literal["omniparser"] = "omniparser"
    preload_on_startup: bool = False
    device: Literal["cpu", "cuda"] = "cuda"
    gpu_uuid: str = ""
    python_path: str = "~/.local/share/adam/omniparser-runtime/bin/python"
    model_path: str = "~/.local/share/adam/models/omniparser-yolov8n.pt"
    confidence_threshold: float = 0.05
    max_regions: int = 60
    timeout_seconds: float = 15.0

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value):
        if not 0.0 < value < 1.0:
            raise ValueError("Computer vision confidence threshold must be between 0 and 1.")
        return value

    @field_validator("max_regions")
    @classmethod
    def validate_max_regions(cls, value):
        if not 1 <= value <= 200:
            raise ValueError("Computer vision max_regions must be between 1 and 200.")
        return value

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
    provider_only: list[str] = Field(default_factory=list)
    allow_provider_fallbacks: bool = True
    num_ctx: int = 65536
    temperature: float = 0.1
    think: Union[bool, str] = False
    disable_reasoning_for_tool_free: bool = False
    tool_free_reasoning_effort: (
        Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] | None
    ) = None
    max_tool_rounds: int = Field(default=64, ge=1, le=256)
    request_timeout_seconds: float = Field(default=45.0, ge=1.0, le=120.0)
    vision_request_timeout_seconds: float = Field(default=90.0, ge=1.0, le=120.0)

    @model_validator(mode="after")
    def validate_tool_free_reasoning_options(self):
        if (
            self.disable_reasoning_for_tool_free
            and self.tool_free_reasoning_effort is not None
        ):
            raise ValueError(
                "disable_reasoning_for_tool_free and tool_free_reasoning_effort "
                "cannot both be configured"
            )
        return self

class ExecutionConfig(BaseModel):
    workspace_dir: str = "~/workspace"
    downloads_dir: str = "~/Downloads"
    jobs_log_dir: str = "~/.local/state/adam/jobs"
    max_log_mb: int = 50
    preferred_video_codec: str = "av1"

class TelemetryConfig(BaseModel):
    enabled: bool = False
    path: str = "~/.local/state/adam/telemetry/events.jsonl"

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
    idea_routing: IdeaRoutingConfig = Field(default_factory=IdeaRoutingConfig)
    browser_navigation: BrowserNavigationConfig = Field(default_factory=BrowserNavigationConfig)
    computer_control: ComputerControlConfig = Field(default_factory=ComputerControlConfig)
    computer_vision: ComputerVisionConfig = Field(default_factory=ComputerVisionConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
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
