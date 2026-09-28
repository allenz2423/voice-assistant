import asyncio
import re


def sanitize_confirmation_speech(prompt_text: str, summary: str = "") -> str:
    """Strips hex addresses, parenthetical dumps, PIDs, and yapping from spoken confirmations.
    Returns an ultra-concise verbal question suitable for text-to-speech."""
    if not prompt_text:
        return (summary.strip().rstrip(".") + "?") if summary else "Confirm action?"

    # 1. Strip parenthesized technical details (hex addresses, PIDs, process addresses, paths)
    cleaned = re.sub(r"\s*\([^)]*(?:0x[0-9a-fA-F]+|pid|process|address|/)[^)]*\)", "", prompt_text, flags=re.IGNORECASE)
    # 2. Strip standalone hex addresses and memory pointers
    cleaned = re.sub(r"\b0x[0-9a-fA-F]+\b", "", cleaned)
    # 3. Strip markdown syntax
    cleaned = re.sub(r"[`*_~]", "", cleaned)
    # 4. Collapse whitespace and fix stray punctuation
    cleaned = re.sub(r"\s+([,.?!])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", cleaned) if s.strip()]
    if not sentences:
        return (summary.strip().rstrip(".") + "?") if summary else "Confirm action?"

    def is_concise(text: str) -> bool:
        words = text.split()
        if len(words) > 12 or len(text) > 75:
            return False
        if re.search(r"(/[\w.-]+){2,}", text):
            return False
        return True

    # If the first sentence is an explicit question (e.g. 'Are you sure you want to kill all Alacritty terminals?'),
    # speak that directly without yapping through subsequent technical explanation sentences.
    if sentences[0].endswith("?") and is_concise(sentences[0]):
        return sentences[0]

    # If the total cleaned prompt is already concise, speak it as is
    if is_concise(cleaned):
        return cleaned

    # If a concise summary is available and the prompt was verbose or path-heavy, formulate a brief question from summary
    if summary:
        s = summary.strip().rstrip(".")
        if not s.endswith("?"):
            s = f"{s}?"
        return s

    # Fallback to the first sentence if concise
    if is_concise(sentences[0]):
        first = sentences[0].rstrip(".")
        if not first.endswith("?"):
            first = f"{first}?"
        return first

    return sentences[0]


class TriStateConfirmationManager:
    """Handles confirmations with active asyncio watchdog timer and word-boundary safety checks."""
    # Strict negation matching evaluated FIRST to prevent false affirmation on negated phrases
    DENY_REGEX = re.compile(r"\b(no|nope|cancel|stop|don't|dont|abort|nevermind|leave it|do not|never|not|negative)\b", re.IGNORECASE)
    AFFIRM_REGEX = re.compile(r"\b(yes|yeah|yep|confirm|proceed|go ahead|sure|do it|ok|okay|please do|sort them|move them|yes please|absolutely|definitely)\b", re.IGNORECASE)
    CLARIFY_REGEX = re.compile(r"\b(what\s+is\s+this|why|repeat|which|wait\b(?!\s+(?:stop|cancel|no))|explain|how)\b", re.IGNORECASE)

    def __init__(self, tts_engine, arbiter, timeout_seconds=10.0):
        self.tts = tts_engine
        self.arbiter = arbiter
        self.timeout_seconds = timeout_seconds
        self.pending_action = None
        self.last_confirmed_action = None
        self.watchdog_task = None

    async def request_confirmation(self, action_payload: dict, prompt_text: str):
        self.pending_action = action_payload
        await self.arbiter.set_state("AWAITING_CONFIRMATION")

        # Flash desktop notification with full technical details (hex addresses, command, process lists)
        notif_title = action_payload.get("summary") or "Confirmation Required"
        notif_body = action_payload.get("details") or prompt_text
        cmd = action_payload.get("command")
        if cmd and cmd not in notif_body:
            notif_body = f"{notif_body}\n\nCommand: {cmd}"

        try:
            from src.tools.desktop import show_desktop_notification
            await asyncio.to_thread(show_desktop_notification, str(notif_title), str(notif_body), "normal")
        except Exception as e:
            print(f"[Confirmation] Notification dispatch failed: {e}", flush=True)

        # Sanitize prompt for TTS so speech is ultra-concise and never reads hex addresses or technical dumps
        spoken_prompt = sanitize_confirmation_speech(prompt_text, summary=action_payload.get("summary", ""))
        print(f"[Adam] Response: {spoken_prompt}", flush=True)
        await self.tts.speak_async(spoken_prompt)

        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
        self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))

    async def _expiry_watchdog(self, duration: float):
        """Active timer that breaks state deadlocks if the user says nothing."""
        try:
            await asyncio.sleep(duration)
            if self.pending_action:
                await self._cancel_confirmation("Confirmation timed out after 10 seconds of silence.")
        except asyncio.CancelledError:
            pass

    async def evaluate_response(self, user_transcript: str) -> tuple[str, str | None]:
        """Classifies response with strict word boundaries and negation priority."""
        if not self.pending_action:
            return "EXPIRED", None

        text = user_transcript.strip()

        # 1. Negative triggers evaluated FIRST (prevents 'no, do not do it' being affirmed by 'do it')
        if self.DENY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.pending_action = None
            await self._cancel_confirmation("Action cancelled.")

            # Check if user chained a new command after cancellation (e.g. "Wait, stop that, what time is it in Tokyo?")
            subsequent_cmd = re.sub(r"^(wait,?\s*)?(stop|cancel|no|nevermind|don'?t do that|do not do it)\b\s*(that,?)?\s*", "", text, flags=re.IGNORECASE).strip()
            return "DENY", (subsequent_cmd if subsequent_cmd else None)

        # 2. Affirmative triggers (guaranteed no negation words present)
        if self.AFFIRM_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.last_confirmed_action = self.pending_action
            self.pending_action = None
            await self.arbiter.set_state("PROCESSING_REACT")
            return "AFFIRM", None

        # 3. Explicit clarification request
        if self.CLARIFY_REGEX.search(text):
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.watchdog_task = asyncio.create_task(self._expiry_watchdog(self.timeout_seconds))
            
            explanation = f"I am waiting to execute: {self.pending_action.get('summary', 'this command')}. Should I proceed?"
            print(f"[Adam] Response: {explanation}", flush=True)
            await self.tts.speak_async(explanation)
            return "CLARIFY", None

        # 4. If the user spoke a multi-word phrase that isn't a yes/no answer, treat it as a new command!
        words = text.split()
        if len(words) >= 3:
            if self.watchdog_task:
                self.watchdog_task.cancel()
            self.pending_action = None
            print(f"[Confirmation] User interrupted with new command: '{text}'", flush=True)
            await self.arbiter.set_state("IDLE_LISTENING")
            return "NEW_COMMAND", text

        # 5. Unrecognized short utterance -> ask for clarification
        print("[Adam] Response: Please answer yes or no.", flush=True)
        await self.tts.speak_async("Please answer yes or no.")
        return "CLARIFY", None

    async def _cancel_confirmation(self, message: str):
        self.pending_action = None
        print(f"[Adam] Response: {message}", flush=True)
        await self.tts.speak_async(message)
        await self.arbiter.set_state("IDLE_LISTENING")
