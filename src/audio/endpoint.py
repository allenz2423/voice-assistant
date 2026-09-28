"""Semantic Endpoint Analyzer for Shin Voice Assistant.

Determines whether a streaming speech utterance represents a complete linguistic command,
a dangling / hesitant thought, or a neutral state, dynamically adjusting silence cutoff
durations between 350ms (instant snappy response) and 1800ms (patient pause).
"""

from __future__ import annotations

import re
from enum import Enum


class EndpointState(Enum):
    COMPLETE = "complete"      # Utterance is grammatically/actionably complete -> shorten silence (~0.35s)
    INCOMPLETE = "incomplete"  # Utterance ends on dangling connector/preposition -> lengthen silence (~1.75s)
    NEUTRAL = "neutral"        # Ambiguous / mid-length phrase -> standard silence (~0.85s)


class SemanticEndpointer:
    """Linguistic and syntactic endpoint analyzer for real-time streaming transcripts."""

    DEFAULT_COMPLETE_SILENCE = 0.35
    DEFAULT_INCOMPLETE_SILENCE = 1.75
    DEFAULT_NEUTRAL_SILENCE = 0.85

    # Connectors, prepositions, conjunctions, and auxiliaries that strongly indicate more speech is coming
    DANGLING_END_WORDS = {
        # Prepositions
        "to", "for", "in", "at", "on", "with", "from", "about", "of", "into", "onto",
        "by", "through", "during", "before", "after", "towards", "toward", "under", "over",
        # Conjunctions
        "and", "or", "but", "because", "if", "so", "although", "while", "since", "that",
        "than", "yet", "nor", "though", "unless",
        # Determiners & Articles
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her",
        "their", "our", "its", "some", "any", "which", "whose",
        # Question / Relative starters dangling at end
        "who", "what", "where", "when", "why", "how",
        # Verbal auxiliaries & dangling modals
        "could", "can", "would", "will", "should", "might", "may", "shall",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        # Hesitation & Filler words
        "uh", "um", "er", "ah", "like", "well", "wait", "hold",
        # Incomplete search / command starters
        "search", "find", "google", "lookup", "check", "tell", "show", "open", "launch"
    }

    # Two-word dangling phrases (e.g. "search for", "could you", "set a")
    DANGLING_PHRASES = {
        "could you", "can you", "would you", "will you", "please to", "i want", "i need",
        "search for", "search up", "look up", "find me", "tell me", "show me", "check on",
        "set a", "set the", "remind me", "go to", "switch to", "move to", "send to",
        "turn on", "turn off", "type in", "write down", "write a", "put on",
        "more than", "less than", "between the", "next to", "ahead of", "because of",
        "hold on", "wait for", "wait a"
    }

    # Actionable standalone commands that are definitely complete (even if very short)
    COMPLETE_STANDALONE_COMMANDS = {
        # Media controls
        "stop", "pause", "resume", "play", "next", "previous", "mute", "unmute",
        "pause music", "stop music", "resume music", "play music", "next track", "previous track",
        "volume up", "volume down", "mute audio", "unmute audio",
        # Desktop & Window controls
        "close window", "close tab", "close this tab", "close this window", "close it",
        "show desktop", "minimize all", "lock screen", "overview", "toggle floating",
        "maximize", "fullscreen", "workspace one", "workspace two", "workspace three",
        # Common instant queries
        "what time is it", "what's the time", "current time",
        "what day is it", "what's the date", "what is today's date",
        "what's the weather", "weather forecast", "is it raining",
        "how much battery", "battery level", "battery status",
        "what's my cpu", "cpu usage", "system status",
        "list windows", "list open windows", "what's open",
        # Confirmations
        "yes", "no", "yeah", "nope", "cancel", "never mind", "nevermind",
        "confirm", "proceed", "go ahead", "do it", "sure", "ok", "okay",
        "thank you", "thanks", "done", "goodbye", "bye"
    }

    PUNCT_SPLIT = re.compile(r"[,.!?:;\"'\`\s]+")

    def __init__(
        self,
        complete_silence: float = DEFAULT_COMPLETE_SILENCE,
        incomplete_silence: float = DEFAULT_INCOMPLETE_SILENCE,
        neutral_silence: float = DEFAULT_NEUTRAL_SILENCE,
    ):
        self.complete_silence = complete_silence
        self.incomplete_silence = incomplete_silence
        self.neutral_silence = neutral_silence

    def clean_text(self, text: str) -> str:
        """Strips leading wake word prefixes or punctuation to analyze the core request."""
        clean = text.strip()
        # Strip trailing punctuation
        clean = re.sub(r"[\"'\‘\’\“\”\`]+$", "", clean).strip()
        return clean

    def analyze(self, text: str, wake_word: str = "") -> tuple[EndpointState, float]:
        """Analyzes a partial transcript and returns (EndpointState, target_silence_duration_seconds)."""
        if not text or not text.strip():
            return EndpointState.NEUTRAL, self.neutral_silence

        raw = text.strip()
        lower = raw.lower()

        # Strip wake word prefix if provided
        if wake_word:
            wake_clean = wake_word.strip().lower()
            if lower.startswith(wake_clean):
                lower = lower[len(wake_clean):].lstrip(" ,:;!-.")
            elif wake_clean.startswith("hey ") and lower.startswith(wake_clean[4:]):
                lower = lower[len(wake_clean[4:]):].lstrip(" ,:;!-.")

        lower = re.sub(r"[.!?。！？]+$", "", lower).strip()
        if not lower:
            # Only wake word was spoken so far -> wait for command
            return EndpointState.INCOMPLETE, self.incomplete_silence

        # Check explicit standalone complete commands
        if lower in self.COMPLETE_STANDALONE_COMMANDS:
            return EndpointState.COMPLETE, self.complete_silence

        words = [w for w in self.PUNCT_SPLIT.split(lower) if w]
        if not words:
            return EndpointState.NEUTRAL, self.neutral_silence

        last_word = words[-1]
        last_two = " ".join(words[-2:]) if len(words) >= 2 else ""

        # 1. High-confidence INCOMPLETE check (dangling phrase or connector)
        if last_two in self.DANGLING_PHRASES:
            return EndpointState.INCOMPLETE, self.incomplete_silence

        if last_word in self.DANGLING_END_WORDS:
            return EndpointState.INCOMPLETE, self.incomplete_silence

        # 2. Check for punctuation at the end of the original raw text
        if raw.endswith((".", "?", "!", "。", "？", "！")):
            # If the user completed an explicit sentence ending with punctuation
            if last_word not in self.DANGLING_END_WORDS:
                return EndpointState.COMPLETE, self.complete_silence

        # 3. High-confidence COMPLETE patterns (imperative verbs + nouns with >= 3 words)
        # e.g., "open firefox browser", "set volume to fifty percent", "what is the capital of france"
        if len(words) >= 3:
            # If it's a complete query starting with question word and ending in a valid noun
            if words[0] in ("what", "what's", "where", "how", "who", "when") and last_word not in self.DANGLING_END_WORDS:
                return EndpointState.COMPLETE, self.complete_silence

            # Common complete verb-noun phrases
            if words[0] in ("open", "launch", "close", "focus", "switch", "show", "search") and last_word not in self.DANGLING_END_WORDS:
                # If command is "search <query>" where query has >= 2 words (e.g. "search santal 33")
                if words[0] == "search" and len(words) >= 2:
                    return EndpointState.COMPLETE, self.complete_silence
                return EndpointState.COMPLETE, self.complete_silence

        # Default fallback
        return EndpointState.NEUTRAL, self.neutral_silence
