import re
from collections import deque
import numpy as np
import openwakeword
from openwakeword.model import Model as OWWModel

class WakeWordDetector:
    """Always-on wake word engine supporting openWakeWord ONNX models and custom phrase VAD spotting."""
    INTERRUPT_KEYWORDS = re.compile(r"\b(stop|cancel|shut up|wait|pause|quiet)\b", re.IGNORECASE)

    def __init__(self, wake_word="hey_jarvis", threshold=0.50, aliases=None, energy_floor=0.0005):
        self.raw_wake_word = wake_word.strip()
        self.wake_word = self.raw_wake_word.replace(" ", "_").lower()
        self.threshold = threshold
        self.energy_floor = max(0.0, float(energy_floor))
        self._quiet_context_max_samples = int(16000 * 2.1)
        self._quiet_context = deque()
        self._quiet_context_samples = 0
        self._model_frames_processed = 0
        self.aliases = aliases or []
        self.model = None
        self.model_key = None
        self.is_custom_mode = False
        self.custom_regex = None
        self.explicit_wake_regex = self._build_explicit_wake_regex(self.raw_wake_word, self.aliases)
        self._load_model()

    def _load_model(self):
        try:
            import glob
            import os
            print(f"[Wake] Searching for openWakeWord model matching '{self.wake_word}'...")
            available_paths = list(openwakeword.get_pretrained_model_paths())
            available_paths.extend(glob.glob("assets/models/*.onnx"))
            selected_path = None
            for p in available_paths:
                if self.wake_word in os.path.basename(p).lower():
                    selected_path = p
                    break

            if selected_path:
                self.model = OWWModel(wakeword_model_paths=[selected_path])
                self.model_key = list(self.model.models.keys())[0] if self.model.models else self.wake_word
                self.is_custom_mode = False
                behavior = "low-energy frames are gated" if self.energy_floor > 0 else "energy gating disabled"
                print(f"[Wake] openWakeWord ONNX model loaded for key: '{self.model_key}' ({behavior}).")
            else:
                self.is_custom_mode = True
                self.custom_regex = self._build_wake_regex(self.raw_wake_word, self.aliases)
                print(f"[Wake] Custom wake word '{self.raw_wake_word}' activated using VAD + Whisper keyphrase spotting.")
        except Exception as e:
            print(f"[Wake] Notice: Using custom phrase spotting for '{self.raw_wake_word}' ({e}).")
            self.is_custom_mode = True
            self.custom_regex = self._build_wake_regex(self.raw_wake_word, self.aliases)

    FILLERS = {"uh", "um", "er", "ah", "so", "okay", "ok", "well", "all", "right", "alright"}
    GREETINGS = {"hey", "hi", "hello", "yo"}
    ALLOWED_PREFIX_WORDS = FILLERS | GREETINGS
    PURE_HESITATIONS = {"uh", "um", "er", "ah"}

    PUNCT_CHARS = r",.!?:;\"'\‘\’\“\”\`，、。！？；：「」…—–\-\s~/\#@*()"
    PUNCT_SPLIT = re.compile(rf"[{PUNCT_CHARS}]+")
    LEADING_PUNCT = re.compile(rf"^[{PUNCT_CHARS}]+")
    TRAILING_PUNCT = re.compile(rf"[\"'\‘\’\“\”\`]+$")

    @classmethod
    def _build_wake_regex(cls, raw_wake_word: str, aliases: list[str] = None) -> re.Pattern:
        raw_lower = raw_wake_word.lower().strip()
        if raw_lower.startswith("hey "):
            base_name = raw_lower[4:].strip()
        else:
            base_name = raw_lower

        names = set()
        names.add(base_name)
        if raw_lower != base_name:
            names.add(raw_lower)

        # Automatic phonetic sound-alike rules for words with alternate spellings (e.g. f <-> ph)
        if "f" in base_name:
            names.add(base_name.replace("f", "ph"))
        if "ph" in base_name:
            names.add(base_name.replace("ph", "f"))

        # User-configured custom aliases (e.g. from config.yaml wake.aliases)
        if aliases:
            for a in aliases:
                a_clean = str(a).lower().strip()
                if a_clean.startswith("hey "):
                    names.add(a_clean[4:].strip())
                names.add(a_clean)

        patterns = []
        punct_sep = r"[\s,，、—–\.\-\'\’\"]*"

        for n in sorted(names, key=len, reverse=True):
            parts = re.findall(r"[\u4e00-\u9fff]|[\u3040-\u30ff]+|[a-zA-Z0-9]+", n)
            if not parts:
                patterns.append(re.escape(n))
                continue

            escaped_parts = []
            for p in parts:
                if any(ord(c) > 127 for c in p):
                    escaped_parts.append(re.escape(p))
                else:
                    escaped_parts.append(rf"\b{re.escape(p)}\b" if len(parts) == 1 else re.escape(p))

            inner = punct_sep.join(escaped_parts)
            if not any(ord(c) > 127 for c in n):
                patterns.append(rf"\b{inner}\b")
            else:
                patterns.append(inner)

        return re.compile("|".join(patterns), re.IGNORECASE)

    @classmethod
    def _build_explicit_wake_regex(cls, raw_wake_word: str, aliases: list[str] = None) -> re.Pattern:
        """Build interruption patterns that require the explicit Hey prefix."""
        phrases = [raw_wake_word, *(aliases or [])]
        patterns = []
        punct_sep = r"[\s,，、—–\.\-\'\’\"]*"
        for phrase in phrases:
            clean = re.sub(r"[_-]+", " ", str(phrase or "").lower().strip())
            if not clean:
                continue
            if not clean.startswith("hey "):
                clean = f"hey {clean}"
            parts = re.findall(r"[\u4e00-\u9fff]|[\u3040-\u30ff]+|[a-zA-Z0-9]+", clean)
            if parts:
                inner = punct_sep.join(re.escape(part) for part in parts)
                patterns.append(rf"(?<!\w){inner}(?!\w)")
        return re.compile("|".join(dict.fromkeys(patterns)) or r"(?!x)x", re.IGNORECASE)

    @classmethod
    def _check_prefix_valid(cls, prefix_str: str) -> bool:
        clean = cls.PUNCT_SPLIT.sub(" ", prefix_str).strip().lower()
        if not clean:
            return True
        for w in clean.split():
            if w in cls.ALLOWED_PREFIX_WORDS:
                continue
            temp = w
            all_sub = True
            while temp:
                matched_sub = False
                for cand in sorted(cls.ALLOWED_PREFIX_WORDS, key=len, reverse=True):
                    if temp.startswith(cand):
                        temp = temp[len(cand):]
                        matched_sub = True
                        break
                if not matched_sub:
                    all_sub = False
                    break
            if not all_sub:
                return False
        return True

    @classmethod
    def _has_explicit_greeting_before_match(cls, prefix_str: str) -> bool:
        """Allow an explicit greeting + wake name later in a self-corrected utterance."""
        words = cls.PUNCT_SPLIT.sub(" ", prefix_str).strip().lower().split()
        for index in range(len(words) - 1, -1, -1):
            if words[index] in cls.GREETINGS and all(
                word in cls.ALLOWED_PREFIX_WORDS for word in words[index:]
            ):
                return True
        return False

    def predict(self, audio_chunk_16k: np.ndarray) -> tuple[bool, float]:
        """Processes a 16kHz audio chunk for ONNX wake word trigger."""
        if self.model is None or self.is_custom_mode:
            return False, 0.0
        if len(audio_chunk_16k) == 0:
            return False, 0.0

        if audio_chunk_16k.dtype == np.float32:
            energy = float(np.sqrt(np.mean(audio_chunk_16k**2))) if len(audio_chunk_16k) else 0.0
            int16_chunk = (audio_chunk_16k * 32767.0).astype(np.int16)
        else:
            int16_chunk = audio_chunk_16k.astype(np.int16)
            normalized = int16_chunk.astype(np.float32) / 32768.0
            energy = float(np.sqrt(np.mean(normalized**2))) if len(normalized) else 0.0

        # Fill the model's five-frame prediction buffer after startup/reset before
        # gating silence. Low-energy frames are cached to restore feature context.
        if self.energy_floor > 0 and self._model_frames_processed >= 5 and energy < self.energy_floor:
            self._remember_quiet_audio(int16_chunk)
            return False, 0.0

        try:
            if self._quiet_context:
                for quiet_chunk in self._quiet_context:
                    self.model.preprocessor(quiet_chunk)
                self._quiet_context.clear()
                self._quiet_context_samples = 0
            prediction = self.model.predict(int16_chunk)
            self._model_frames_processed += 1
            score = prediction.get(self.model_key, 0.0)
            return score >= self.threshold, float(score)
        except Exception:
            return False, 0.0

    def _remember_quiet_audio(self, chunk: np.ndarray) -> None:
        """Retain recent silence so the ONNX feature window is current on wake onset."""
        values = np.asarray(chunk, dtype=np.int16).copy()
        if len(values) > self._quiet_context_max_samples:
            values = values[-self._quiet_context_max_samples:]
        self._quiet_context.append(values)
        self._quiet_context_samples += len(values)
        while self._quiet_context and self._quiet_context_samples > self._quiet_context_max_samples:
            oldest = self._quiet_context.popleft()
            self._quiet_context_samples -= len(oldest)

    def match_custom_wake_word(self, text: str) -> tuple[bool, str]:
        """Find the last addressed wake phrase and return only the command after it.

        Scanning the whole transcript lets a user correct or restart a request with
        a later wake phrase address without accidentally sending the abandoned first clause.
        Bare-name aliases remain anchored to the utterance or a sentence boundary.
        """
        if not self.is_custom_mode or not self.custom_regex or not text:
            return False, text

        clean_text = text.strip()

        valid_matches = []
        for match in self.custom_regex.finditer(clean_text):
            start, end = match.span()
            prefix_str = clean_text[:start]
            sentence_prefix = re.split(r"[.!?。！？\n]+", prefix_str)[-1]

            if (
                self._check_prefix_valid(prefix_str)
                or self._check_prefix_valid(sentence_prefix)
                or self._has_explicit_greeting_before_match(prefix_str)
            ):
                valid_matches.append(match)

        if valid_matches:
            # If the user repeats/restarts the address, the latest one begins the
            # intended command; earlier corrected clauses should be discarded.
            match = valid_matches[-1]
            remaining = clean_text[match.end():]
            remaining = self.LEADING_PUNCT.sub("", remaining).strip()
            remaining = self.TRAILING_PUNCT.sub("", remaining).strip()

            # Strip leading pure hesitations from command
            while True:
                parts = self.PUNCT_SPLIT.split(remaining, maxsplit=1)
                if parts and parts[0].lower() in self.PURE_HESITATIONS:
                    remaining = remaining[len(parts[0]):]
                    remaining = self.LEADING_PUNCT.sub("", remaining).strip()
                else:
                    break
            return True, remaining

        return False, text

    def match_explicit_wake_word(self, text: str) -> tuple[bool, str]:
        """Match an interruption only when the transcript includes the Hey prefix."""
        if not text:
            return False, text
        matches = list(self.explicit_wake_regex.finditer(text))
        if not matches:
            return False, text
        remainder = text[matches[-1].end():].strip()
        remainder = self.LEADING_PUNCT.sub("", remainder).strip()
        remainder = self.TRAILING_PUNCT.sub("", remainder).strip()
        return True, remainder

    def is_interrupt_phrase(self, text: str) -> bool:
        """Checks if a short transcribed utterance is an immediate barge-in interrupt command."""
        if not text:
            return False
        return bool(self.INTERRUPT_KEYWORDS.search(text.strip()))

    def reset(self):
        if self.model:
            self.model.reset()
        self._quiet_context.clear()
        self._quiet_context_samples = 0
        self._model_frames_processed = 0
