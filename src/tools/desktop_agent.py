"""A general focused-window desktop loop driven by TypeSafe Jev decisions."""

from __future__ import annotations

import re
import time

from src.tools.computer_control import ComputerController
from src.tools.jev_decision import JevDecisionClient


_CONSEQUENTIAL_CONTROL = re.compile(
    r"\b(send|submit|purchase|buy|checkout|place order|delete|remove|publish|post|confirm|transfer)\b",
    re.IGNORECASE,
)
_CREDENTIAL_CONTEXT = re.compile(
    r"\b(password|passcode|one[- ]time code|verification code|secret key|private key|security token)\b",
    re.IGNORECASE,
)
_QUOTED_TARGET = re.compile(r'["“”‘’\']([^"“”‘’\']{2,160})["“”‘’\']')
_TARGET_STOPWORDS = {"a", "an", "and", "are", "at", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


class DesktopComputerAgent:
    """Select safe operations and OCR targets for the focused desktop window."""

    def __init__(
        self,
        controller: ComputerController,
        decisions: JevDecisionClient,
        max_steps: int = 12,
        repeat_limit: int = 2,
        wait_seconds: float = 0.6,
    ) -> None:
        self.controller = controller
        self.decisions = decisions
        self.max_steps = max(1, min(int(max_steps), 20))
        self.repeat_limit = max(1, min(int(repeat_limit), 4))
        self.wait_seconds = min(max(float(wait_seconds), 0.1), 3.0)

    def run(self, goal: str, text_to_enter: str = "", max_steps: int | None = None) -> str:
        goal = " ".join((goal or "").split())
        if not goal:
            return "Desktop task needs a goal."
        if len(goal) > 2000 or len(text_to_enter) > self.controller.max_text_length:
            return "Desktop task text exceeded its configured limit."
        max_steps = self.max_steps if max_steps is None else max(1, min(int(max_steps), self.max_steps))

        result = self.controller.run(action="inspect", scope="window")
        trace = [_compact_ocr_output(result.message)]
        if "Could not capture" in result.message or not self.controller.snapshot_id:
            return "\n".join([*trace, "DESKTOP_TASK_STATUS: INCOMPLETE — no usable screen observation."])

        previous_screen = self._screen_signature()
        unchanged_count = 0
        repeated: dict[tuple[str, str], int] = {}
        status = "INCOMPLETE"
        for step in range(1, max_steps + 1):
            decision = self.decisions.decide_next(
                goal,
                self.controller._ocr_state,
                self.controller._ocr_regions,
                text_to_enter,
            )
            trace.append(f"Step {step}: {decision.message}")
            operation = decision.operation
            if operation == "BLOCKED":
                trace.append("Stopped without another input action.")
                break
            if operation in {"CLICK", "TYPE_TEXT"}:
                region = decision.region
                if region is None:
                    trace.append("Stopped because Jev did not select a current OCR target.")
                    break
                if not _matches_quoted_target(goal, region.text):
                    trace.append(
                        f"Stopped before clicking {region.ref}: its OCR text does not match the explicitly named target."
                    )
                    break
                if _CONSEQUENTIAL_CONTROL.search(region.text):
                    trace.append(f"Stopped before clicking {region.ref}: the OCR label may submit or change external data.")
                    break
                if operation == "TYPE_TEXT" and _CREDENTIAL_CONTEXT.search(self.controller._ocr_state):
                    trace.append("Stopped because the visible screen appears to contain a credential prompt.")
                    break
                signature = (operation, region.ref)
                repeated[signature] = repeated.get(signature, 0) + 1
                if repeated[signature] > self.repeat_limit:
                    trace.append("Stopped after Jev repeated the same target without progress.")
                    break
                result = self.controller.run(
                    action="click",
                    snapshot_id=self.controller.snapshot_id,
                    ocr_region_ref=region.ref,
                )
                trace.append(_compact_ocr_output(result.message))
                if operation == "TYPE_TEXT":
                    exact_text = text_to_enter
                    text_to_enter = ""  # Never autonomously repeat user-supplied text.
                    result = self.controller.run(
                        action="type",
                        snapshot_id=self.controller.snapshot_id,
                        text=exact_text,
                    )
                    trace.append(_compact_ocr_output(result.message))
            elif operation in {"SCROLL_UP", "SCROLL_DOWN"}:
                result = self.controller.run(
                    action="scroll",
                    snapshot_id=self.controller.snapshot_id,
                    direction="up" if operation == "SCROLL_UP" else "down",
                    amount=3,
                )
                trace.append(_compact_ocr_output(result.message))
            elif operation == "WAIT":
                time.sleep(self.wait_seconds)
                result = self.controller.run(action="inspect", snapshot_id=self.controller.snapshot_id)
                trace.append(_compact_ocr_output(result.message))

            done, completion_message = self.decisions.check_goal_done(
                goal,
                self.controller._ocr_state,
            )
            trace.append(completion_message)
            if done is True:
                trace.append("Jev reports the requested end state is visible; Adam must independently verify it.")
                status = "COMPLETION_CANDIDATE"
                break
            if done is None:
                trace.append("Stopped because the completion check was inconclusive; inspect the fresh screen before another action.")
                break

            current_screen = self._screen_signature()
            if current_screen == previous_screen:
                unchanged_count += 1
            else:
                unchanged_count = 0
            if unchanged_count >= self.repeat_limit:
                trace.append("Stopped because the OCR state did not change after repeated actions.")
                break
            previous_screen = current_screen
        else:
            trace.append(f"Stopped at the {max_steps}-step safety limit.")

        trace.append("Latest OCR state:\n" + _compact_ocr_output(self.controller._ocr_state, limit=24))
        trace.append(f"DESKTOP_TASK_STATUS: {status} — Adam must inspect the current state before claiming completion or choosing another route.")
        return "\n".join(trace)

    def _screen_signature(self) -> str:
        return "\n".join(region.text.casefold() for region in self.controller._ocr_regions)


# Backward-compatible aliases for the original OCR proof of concept.
KevComputerAgent = DesktopComputerAgent


def _matches_quoted_target(goal: str, region_text: str) -> bool:
    """Require OCR evidence for an explicitly quoted target before clicking it."""
    quoted = _QUOTED_TARGET.findall(goal or "")
    if not quoted:
        return True
    target = quoted[-1]
    terms = {
        token.casefold()
        for token in re.findall(r"[\w]+", target, flags=re.UNICODE)
        if token.casefold() not in _TARGET_STOPWORDS
    }
    if not terms:
        terms = {token.casefold() for token in re.findall(r"[\w]+", target, flags=re.UNICODE)}
    visible = {token.casefold() for token in re.findall(r"[\w]+", region_text or "", flags=re.UNICODE)}
    required = max(1, (len(terms) * 3 + 4) // 5)
    return len(terms & visible) >= required


def _compact_ocr_output(message: str, limit: int = 10) -> str:
    """Keep returned OCR evidence readable while preserving the full internal state for Jev."""
    marker = "OCR text regions ("
    start = message.find(marker)
    if start < 0:
        return message
    prefix = message[:start].rstrip()
    lines = message[start:].splitlines()
    # First line is the heading; subsequent lines are OCR observations.
    shown = lines[1 : limit + 1]
    count = max(0, len(lines) - 1 - len(shown))
    compact = "\n".join([lines[0], *shown])
    if count:
        compact += f"\n… {count} additional OCR regions omitted from this trace."
    return f"{prefix}\n{compact}" if prefix else compact
