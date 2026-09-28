import asyncio
import re

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
        print(f"[Shin] Response: {prompt_text}", flush=True)
        await self.tts.speak_async(prompt_text)

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
            print(f"[Shin] Response: {explanation}", flush=True)
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
        print("[Shin] Response: Please answer yes or no.", flush=True)
        await self.tts.speak_async("Please answer yes or no.")
        return "CLARIFY", None

    async def _cancel_confirmation(self, message: str):
        self.pending_action = None
        print(f"[Shin] Response: {message}", flush=True)
        await self.tts.speak_async(message)
        await self.arbiter.set_state("IDLE_LISTENING")
