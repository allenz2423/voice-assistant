"""TypeSafe Jev Decisions API client for bounded desktop UI choices."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.tools.ocr import OCRRegion


@dataclass(frozen=True)
class JevActionDecision:
    operation: str
    region: OCRRegion | None
    confidence: float
    message: str


class JevDecisionClient:
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "typesafe/jev-1.13",
        api_key: str = "",
        timeout_seconds: float = 20.0,
        min_confidence: float = 0.65,
        opener=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key.strip()
        self.timeout_seconds = min(max(float(timeout_seconds), 0.2), 30.0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self._opener = opener or urllib.request.urlopen
        self._last_target_confidence = 0.0

    def choose_ocr_target(
        self,
        goal: str,
        target_description: str,
        page_state: str,
        regions: list[OCRRegion],
    ) -> tuple[OCRRegion | None, str]:
        self._last_target_confidence = 0.0
        if not regions:
            return None, "OCR found no text targets to choose from."
        # Reserve one of TypeSafe System One's 255 option slots for abstaining.
        choices = regions[:254]
        criteria = {
            region.ref: f"Visible screen text: {region.text}; screen location: {region.describe()}"
            for region in choices
        }
        payload = {
            "model": self.model,
            "state": {
                "user_goal": goal[:2000],
                "requested_target": target_description[:500],
                "current_screen": page_state[:8000],
            },
            "questions": {
                "target": {
                    "type": "choice",
                    "instructions": (
                        "Which single visible text region should be clicked or focused next to make progress toward the user's goal? "
                        "Choose only a region whose visible text and context match the requested target. If none is a safe, "
                        "clear match, choose the option named NONE."
                    ),
                    "criteria": {"NONE": "No visible text region is a clear and safe match.", **criteria},
                }
            },
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return None, f"Jev decision service unavailable: {type(exc).__name__}: {str(exc)[:180]}"
        except Exception as exc:
            return None, f"Jev decision failed: {type(exc).__name__}: {str(exc)[:180]}"

        answer = (((data or {}).get("answers") or {}).get("target") or {})
        selected = str(answer.get("choice") or "").strip()
        confidence = float(answer.get("confidence") or 0.0)
        if selected.upper() == "NONE":
            return None, "Jev found no clear OCR target match."
        region = next((item for item in choices if item.ref.casefold() == selected.casefold()), None)
        if region is None:
            return None, f"Jev returned an unknown OCR target ({selected[:40]})."
        if confidence < self.min_confidence:
            return None, f"Jev target confidence was too low ({confidence:.2f}); no click was made."
        self._last_target_confidence = confidence
        return region, f"Jev selected OCR target {region.ref} (confidence={confidence:.2f})."

    def decide_next(
        self,
        goal: str,
        page_state: str,
        regions: list[OCRRegion],
        text_to_enter: str = "",
    ) -> JevActionDecision:
        """Choose an operation and compatible OCR target in one Jev API request."""
        regions = regions[:254]
        operation_criteria = {
            "CLICK": "Click a visible text-labeled control or link that directly advances the user's goal.",
            "TYPE_TEXT": (
                "Focus a visible text-entry region and enter the exact text supplied by the user."
                if text_to_enter
                else "Unavailable: the user did not provide exact text to enter."
            ),
            "SCROLL_DOWN": "Scroll down because the needed content/control is likely below the visible screen.",
            "SCROLL_UP": "Scroll up because the needed content/control is likely above the visible screen.",
            "WAIT": "Wait briefly because the current page is visibly loading or changing.",
            "BLOCKED": "The goal cannot safely progress from the visible OCR state, or a required choice is ambiguous.",
        }
        target_criteria = {
            region.ref: f"Visible screen text: {region.text}; location: {region.describe()}"
            for region in regions
        }
        payload = {
            "model": self.model,
            "state": {
                "user_goal": goal[:2000],
                "exact_user_text_to_enter": text_to_enter[:1000],
                "current_screen_ocr": page_state,
            },
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": (
                        "Choose exactly one safest next operation toward the user's goal. Do not repeat an action that "
                        "already left the visible OCR state unchanged. Never choose an operation that submits, sends, "
                        "purchases, deletes, publishes, or confirms external changes. Completion is checked separately "
                        "after this action; choose BLOCKED if no useful safe action remains."
                    ),
                    "criteria": operation_criteria,
                },
                "target": {
                    "type": "choice",
                    "instructions": (
                        "Choose the visible text region needed for the requested operation and goal. "
                        "If no region is a clear match, choose NONE."
                    ),
                    "criteria": {"NONE": "No clear, safe target is visible.", **target_criteria},
                },
            },
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return JevActionDecision("BLOCKED", None, 0.0, f"Jev decision service unavailable: {type(exc).__name__}: {str(exc)[:180]}")
        except Exception as exc:
            return JevActionDecision("BLOCKED", None, 0.0, f"Jev decision failed: {type(exc).__name__}: {str(exc)[:180]}")

        answers = (data or {}).get("answers") or {}
        operation_answer = answers.get("operation") or {}
        operation = str(operation_answer.get("choice") or "BLOCKED").upper()
        operation_confidence = float(operation_answer.get("confidence") or 0.0)
        if operation not in operation_criteria:
            return JevActionDecision("BLOCKED", None, operation_confidence, f"Jev returned unsupported operation {operation[:40]}.")
        if operation_confidence < self.min_confidence:
            return JevActionDecision("BLOCKED", None, operation_confidence, f"Jev operation confidence was too low ({operation_confidence:.2f}).")
        if operation == "TYPE_TEXT" and not text_to_enter:
            return JevActionDecision("BLOCKED", None, operation_confidence, "Jev selected text entry but no exact user-provided text is available.")

        region = None
        if operation in {"CLICK", "TYPE_TEXT"}:
            target_answer = answers.get("target") or {}
            target = str(target_answer.get("choice") or "").strip()
            target_confidence = float(target_answer.get("confidence") or 0.0)
            if target.upper() == "NONE":
                return JevActionDecision("BLOCKED", None, operation_confidence, "Jev found no clear OCR target match.")
            region = next((item for item in regions if item.ref.casefold() == target.casefold()), None)
            if region is None:
                return JevActionDecision("BLOCKED", None, operation_confidence, f"Jev returned an unknown OCR target ({target[:40]}).")
            if target_confidence < self.min_confidence:
                return JevActionDecision("BLOCKED", None, operation_confidence, f"Jev target confidence was too low ({target_confidence:.2f}); no click was made.")
        else:
            target_confidence = 0.0
        return JevActionDecision(
            operation,
            region,
            operation_confidence,
            f"Jev selected {operation} (confidence={operation_confidence:.2f})"
            + (f" on OCR target {region.ref} (confidence={target_confidence:.2f})." if region else "."),
        )

    def check_goal_done(self, goal: str, page_state: str) -> tuple[bool | None, str]:
        """Ask a separate binary question against the full fresh OCR state."""
        payload = {
            "model": self.model,
            "state": {
                "user_goal": goal[:2000],
                "current_screen_ocr": page_state,
            },
            "questions": {
                "done": {
                    "type": "choice",
                    "instructions": (
                        "Is the user's requested end state visibly complete now? Answer YES only if the OCR shows "
                        "evidence that the requested outcome has been reached. Answer NO if another requested action "
                        "is still needed or the evidence is insufficient. Do not suggest or perform an action here."
                    ),
                    "criteria": {
                        "YES": "The visible OCR provides evidence that the user's requested outcome is complete.",
                        "NO": "The goal is not visibly complete, or the OCR evidence is insufficient to confirm it.",
                    },
                }
            },
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return None, f"Jev completion check unavailable: {type(exc).__name__}: {str(exc)[:180]}"
        except Exception as exc:
            return None, f"Jev completion check failed: {type(exc).__name__}: {str(exc)[:180]}"

        answer = (((data or {}).get("answers") or {}).get("done") or {})
        choice = str(answer.get("choice") or "").strip().casefold()
        try:
            confidence = float(answer.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if choice not in {"yes", "no"}:
            return None, f"Jev completion check returned an invalid answer ({choice[:40] or 'empty'})."
        if confidence < self.min_confidence:
            return None, f"Jev completion confidence was too low ({confidence:.2f}); stopped for review."
        return choice == "yes", f"Jev completion check: {choice.upper()} (confidence={confidence:.2f})."

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# Keep imports from the previous local-Kev proof of concept working for old
# tests and local scripts. Adam itself uses JevDecisionClient exclusively.
KevActionDecision = JevActionDecision
KevDecisionClient = JevDecisionClient
