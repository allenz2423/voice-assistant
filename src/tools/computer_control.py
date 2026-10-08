"""Screenshot-guided mouse and keyboard control for X11 and Wayland desktops."""

from __future__ import annotations

import os
import re
import json
import math
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.tools.desktop import (
    capture_screenshot_with_origin,
    screenshot_delay_for_focused_window,
    wait_for_application_ready,
)
from src.tools.ocr import OCRRegion, ScreenOCR
from src.tools.desktop_timing import log_duration, timed_stage, timing_operation


_X11_KEYS = {
    "enter": "Return", "tab": "Tab", "escape": "Escape", "backspace": "BackSpace",
    "delete": "Delete", "space": "space", "up": "Up", "down": "Down",
    "left": "Left", "right": "Right", "home": "Home", "end": "End",
    "pageup": "Prior", "pagedown": "Next", "equal": "equal", "minus": "minus",
}
_WAYLAND_KEYS = {
    "enter": "Return", "tab": "Tab", "escape": "Escape", "backspace": "BackSpace",
    "delete": "Delete", "space": "space", "up": "Up", "down": "Down",
    "left": "Left", "right": "Right", "home": "Home", "end": "End",
    "pageup": "Prior", "pagedown": "Next", "equal": "equal", "minus": "minus",
}
_YDOTOOL_KEYCODES = {
    "enter": 28, "tab": 15, "escape": 1, "backspace": 14, "delete": 111,
    "space": 57, "up": 103, "down": 108, "left": 105, "right": 106,
    "home": 102, "end": 107, "pageup": 104, "pagedown": 109, "equal": 13, "minus": 12,
}
_LETTER_KEYCODES = {
    **dict(zip("qwertyuiop", range(16, 26))),
    **dict(zip("asdfghjkl", [30, 31, 32, 33, 34, 35, 36, 37, 38])),
    **dict(zip("zxcvbnm", [44, 45, 46, 47, 48, 49, 50])),
}
_MODIFIER_CODES = {"ctrl": 29, "alt": 56, "shift": 42, "super": 125}
_ALLOWED_COMBOS = {
    "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+y",
    "ctrl+f", "ctrl+l", "ctrl+t", "ctrl+w", "ctrl+s", "ctrl+plus", "ctrl+minus",
    "ctrl+shift+a", "shift+tab", "shift+home", "shift+end", "alt+left", "alt+right",
}


def _serialized(method):
    """Keep snapshot validation, input dispatch, and recapture in one controller turn."""
    def call(self, *args, **kwargs):
        with timing_operation("controller.operation"):
            lock_started = time.perf_counter()
            self._action_lock.acquire()
            log_duration("controller.lock_wait", lock_started)
            try:
                with timed_stage("controller.run"):
                    return method(self, *args, **kwargs)
            finally:
                self._action_lock.release()
    return call


def detect_display_backend(environ: dict[str, str] | None = None) -> str:
    """Prefer the compositor-native protocol when running inside Wayland/XWayland."""
    env = os.environ if environ is None else environ
    session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
    if session_type == "wayland" or env.get("WAYLAND_DISPLAY"):
        return "wayland"
    if session_type == "x11" or env.get("DISPLAY"):
        return "x11"
    return "unsupported"


def coordinate_mode_for_model(model_name: str) -> str:
    """GUI-Owl emits absolute UI locations in a 0–1000 normalized scale."""
    return "normalized_1000" if "gui-owl" in (model_name or "").lower() else "pixels"


def _png_size(image: bytes) -> tuple[int, int]:
    if len(image) >= 24 and image[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(image[16:20], "big"), int.from_bytes(image[20:24], "big")
    raise RuntimeError("Screenshot did not contain a valid PNG image.")


def _screens_visually_unchanged(previous: bytes, current: bytes) -> bool:
    """Compare tiny grayscale previews to flag clicks with no visible response."""
    try:
        from io import BytesIO
        from PIL import Image, ImageChops, ImageStat

        with Image.open(BytesIO(previous)) as before, Image.open(BytesIO(current)) as after:
            if before.size != after.size:
                return False
            before_preview = before.convert("L").resize((32, 18), Image.Resampling.BILINEAR)
            after_preview = after.convert("L").resize((32, 18), Image.Resampling.BILINEAR)
            difference = ImageStat.Stat(ImageChops.difference(before_preview, after_preview)).mean[0]
        # This deliberately cheap pixel check ignores cursor-level changes.
        # Recognized OCR text is compared separately when the caller requested it.
        return difference <= 0.03
    except Exception:
        return False


def _screen_state_unchanged(
    previous_image: bytes,
    current_image: bytes,
    previous_text: tuple[str, ...] | None = None,
    current_text: tuple[str, ...] | None = None,
) -> bool:
    """Use recognized text changes to catch small controls missed by pixel sampling."""
    if previous_text is not None or current_text is not None:
        # If OCR ran on only one frame, the controller cannot establish that the
        # screen stayed the same. Avoid turning missing evidence into a no-change claim.
        if previous_text is None or current_text is None or previous_text != current_text:
            return False
    return _screens_visually_unchanged(previous_image, current_image)


def _normalize_ocr_target(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _match_ocr_region(
    target_text: str,
    regions: list[OCRRegion],
) -> tuple[OCRRegion | None, str]:
    """Resolve a visible text target only when its OCR match is unambiguous."""
    target = _normalize_ocr_target(target_text)
    if not target:
        return None, "target_text must name a visible OCR label."
    if not regions:
        return None, (
            "No OCR labels are available for this screenshot. "
            "Inspect the current window with include_ocr=true, then select a visible label."
        )
    normalized = [(_normalize_ocr_target(region.text), region) for region in regions]
    exact = [region for text, region in normalized if text == target]
    matches = exact or [region for text, region in normalized if target in text]
    if len(matches) == 1:
        return matches[0], f"Matched visible OCR label {matches[0].text!r}."
    if len(matches) > 1:
        labels = ", ".join(repr(region.text) for region in matches[:5])
        return None, f"target_text is ambiguous among visible labels: {labels}."
    return None, f"No visible OCR label matched {target_text!r}. Inspect the latest screen again."


@dataclass
class ComputerControlResult:
    message: str
    screenshot: bytes | None = None
    status: str = "ok"
    dispatched: bool | None = False
    snapshot_id: str = ""
    ocr_regions: list[OCRRegion] = field(default_factory=list, repr=False)
    raw_screenshot: bytes | None = field(default=None, repr=False)


class ComputerController:
    """Executes one bounded input action against the last inspected screenshot."""

    def __init__(
        self,
        enabled: bool = True,
        max_text_length: int = 20000,
        screenshot_fn: Callable[[], bytes | tuple[bytes, tuple[int, int]]] | None = None,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
        environ: dict[str, str] | None = None,
        ensure_wayland_daemon: Callable[[], bool] | None = None,
        coordinate_mode: str = "pixels",
        visual_grounder: Callable[[bytes], tuple[bytes, str]] | None = None,
        screenshot_delay_seconds: float = 0.25,
        browser_screenshot_delay_seconds: float = 3.0,
        ocr_only: bool = False,
        ocr_reader: ScreenOCR | None = None,
        target_selector: Callable[[str, str, str, list[OCRRegion]], tuple[OCRRegion | None, str]] | None = None,
        max_sequence_actions: int = 8,
        max_sequence_text_length: int = 20000,
        sequence_timeout_seconds: float = 45.0,
    ) -> None:
        self.enabled = enabled
        self.max_text_length = max(1, int(max_text_length))
        self._screenshot_fn = screenshot_fn
        self._scope = "window"
        self._capture_target = "focused window"
        self._capture_backend = "injected screenshot provider" if screenshot_fn is not None else "unresolved"
        self._runner = runner or subprocess.run
        self._env = environ
        self._ensure_wayland_daemon_override = ensure_wayland_daemon
        self._visual_grounder = visual_grounder
        self.screenshot_delay_seconds = min(max(float(screenshot_delay_seconds), 0.0), 10.0)
        self.browser_screenshot_delay_seconds = min(max(float(browser_screenshot_delay_seconds), 0.0), 10.0)
        self.ocr_only = bool(ocr_only)
        self._ocr_reader = ocr_reader or (ScreenOCR() if self.ocr_only else None)
        self._target_selector = target_selector
        self.max_sequence_actions = min(max(int(max_sequence_actions), 1), 8)
        self.max_sequence_text_length = min(max(int(max_sequence_text_length), 1), 160000)
        self.sequence_timeout_seconds = min(max(float(sequence_timeout_seconds), 1.0), 120.0)
        self._ocr_regions: list[OCRRegion] = []
        self._ocr_state = ""
        self._include_visual_grounding = False
        self._include_ocr = bool(self.ocr_only)
        if coordinate_mode not in {"pixels", "normalized_1000"}:
            raise ValueError("coordinate_mode must be 'pixels' or 'normalized_1000'.")
        self.coordinate_mode = coordinate_mode
        self._snapshot_id = ""
        self._last_screenshot: bytes | None = None
        self._has_captured_frame = False
        self._width = self._height = 0
        self._origin_x = self._origin_y = 0
        self._capture_scale = (1.0, 1.0)
        self._active_bounds: tuple[int, int, int, int] | None = None
        self._active_identity: str | None = None
        self._snapshot_identity: str | None = None
        self._held_drag_button: str | None = None
        self._held_drag_position: tuple[int, int] | None = None
        self._action_lock = threading.RLock()

    def _wayland_window_adapter(self) -> str:
        env = os.environ if self._env is None else self._env
        desktops = " ".join((env.get(key, "") for key in ("XDG_CURRENT_DESKTOP", "DESKTOP_SESSION"))).casefold()
        if (env.get("HYPRLAND_INSTANCE_SIGNATURE") or "hyprland" in desktops) and self._which("hyprctl"):
            return "hyprland"
        if (env.get("SWAYSOCK") or "sway" in desktops) and self._which("swaymsg"):
            return "sway"
        return "unsupported"

    def _which(self, command: str) -> str | None:
        """Resolve tools from the controller's session PATH, never the host PATH."""
        if self._env is None:
            return shutil.which(command)
        return shutil.which(command, path=self._env.get("PATH", ""))

    def _command(self, command: str) -> str:
        """Return a resolved executable or fail closed for a scoped session."""
        if self._env is None:
            return command
        resolved = self._which(command)
        if not resolved:
            raise RuntimeError(f"{command} is unavailable on the selected desktop session PATH.")
        return resolved

    def _read_active_window_state(self) -> tuple[str | None, tuple[int, int, int, int] | None]:
        try:
            wayland_adapter = self._wayland_window_adapter() if self.backend == "wayland" else ""
            if wayland_adapter == "hyprland":
                active = self._runner(
                    [self._command("hyprctl"), "activewindow", "-j"], capture_output=True, text=True, timeout=2,
                    env=self._env,
                )
                data = json.loads(active.stdout or "{}")
                identity = str(data.get("address") or "") or (
                    f"{data.get('class', '')}:{data.get('title', '')}:"
                    f"{data.get('at', '')}:{data.get('size', '')}"
                )
                identity = f"hyprland:{identity}" if identity.strip(":") else None
                x, y = map(int, data.get("at", [0, 0]))
                width, height = map(int, data.get("size", [0, 0]))
            elif wayland_adapter == "sway":
                result = self._runner(
                    [self._command("swaymsg"), "-t", "get_tree", "-r"],
                    capture_output=True, text=True, timeout=2,
                    env=self._env,
                )
                tree = json.loads(result.stdout or "{}")

                def find_focused(node):
                    if node.get("focused"):
                        return node
                    for child in [*node.get("nodes", []), *node.get("floating_nodes", [])]:
                        found = find_focused(child)
                        if found:
                            return found
                    return None

                node = find_focused(tree) or {}
                rect = node.get("rect") or {}
                identity = f"sway:{node.get('id')}" if node.get("id") is not None else None
                x, y = int(rect.get("x", 0)), int(rect.get("y", 0))
                width, height = int(rect.get("width", 0)), int(rect.get("height", 0))
            elif self.backend == "x11" and self._which("xdotool"):
                active_id = self._runner(
                    [self._command("xdotool"), "getactivewindow"], capture_output=True, text=True, timeout=2,
                    env=self._env,
                )
                window_id = (active_id.stdout or "").strip()
                result = self._runner(
                    [self._command("xdotool"), "getactivewindow", "getwindowgeometry", "--shell"],
                    capture_output=True, text=True, timeout=2,
                    env=self._env,
                )
                values = dict(
                    line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line
                )
                window_id = window_id or values.get("WINDOW", "")
                identity = f"x11:{window_id}" if window_id else None
                x, y = int(values["X"]), int(values["Y"])
                width, height = int(values["WIDTH"]), int(values["HEIGHT"])
            else:
                return None, None
            scale_x, scale_y = self._capture_scale
            left = round((x - self._origin_x) * scale_x)
            top = round((y - self._origin_y) * scale_y)
            scaled_width = round(width * scale_x)
            scaled_height = round(height * scale_y)
            bounds = (left, top, left + scaled_width, top + scaled_height) if width > 0 and height > 0 else None
            return identity, bounds
        except Exception:
            return None, None

    def _read_active_window_bounds(self) -> tuple[int, int, int, int] | None:
        """Backward-compatible geometry accessor."""
        return self._read_active_window_state()[1]

    @property
    def backend(self) -> str:
        return detect_display_backend(self._env)

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    def invalidate_snapshot(self) -> None:
        """Revoke action coordinates after any state change outside this controller."""
        self._snapshot_id = ""
        self._snapshot_identity = None

    @property
    def available(self) -> bool:
        backend = self.backend
        if not self.enabled:
            return False
        if backend == "wayland":
            adapter = self._wayland_window_adapter()
            if self._env is not None:
                # wtype and Hyprland's shortcut API target the compositor named
                # by this environment. ydotool's uinput device is system-wide.
                return adapter != "unsupported" and (bool(self._which("wtype")) or adapter == "hyprland")
            return bool(
                adapter != "unsupported"
                and self._which("ydotool")
                and (self._which("wtype") or self._which("ydotool"))
            )
        if backend == "x11":
            return bool(self._which("xdotool"))
        return False

    def _capture(
        self,
        prefix: str,
        *,
        issue_action_token: bool = True,
        expected_application: str | None = None,
        wait_until_ready: bool = True,
        readiness_timeout_seconds: float | None = 15.0,
    ) -> ComputerControlResult:
        if self._screenshot_fn is None and wait_until_ready:
            with timed_stage("controller.application_ready", expected_application=bool(expected_application)):
                wait_for_application_ready(
                    expected_application, timeout=readiness_timeout_seconds, environ=self._env,
                )
        with timed_stage("controller.window_state_before"):
            before_identity, _ = self._read_active_window_state()
        with timed_stage("controller.capture"):
            captured = (
                self._screenshot_fn() if self._screenshot_fn is not None
                else capture_screenshot_with_origin(
                    self._scope, wait_until_ready=False, environ=self._env,
                )
            )
        if isinstance(captured, tuple):
            image, (self._origin_x, self._origin_y) = captured
            self._capture_scale = getattr(captured, "scale", (1.0, 1.0))
            self._capture_backend = getattr(captured, "backend", "injected screenshot provider")
            self._capture_target = getattr(captured, "target", "") or self._scope
        else:
            image = captured
            self._origin_x = self._origin_y = 0
            self._capture_scale = (1.0, 1.0)
            self._capture_backend = "injected screenshot provider"
            self._capture_target = self._scope
        raw_screenshot = image
        metadata_started = time.perf_counter()
        self._width, self._height = _png_size(image)
        log_duration(
            "controller.image_dimensions", metadata_started,
            width=self._width, height=self._height, image_bytes=len(image),
        )
        with timed_stage("controller.window_state_after"):
            after_identity, self._active_bounds = self._read_active_window_state()
        self._active_identity = after_identity
        if self._scope == "window" and after_identity:
            self._capture_target = f"focused window ({after_identity})"
        if self._scope == "desktop":
            issue_action_token = False
            prefix += " Full-desktop inspection is read-only; inspect the focused window or a verified monitor before input."
        focus_stable = bool(before_identity and before_identity == after_identity)
        issue_action_token = issue_action_token and focus_stable
        if not focus_stable:
            prefix += " Focus could not be confirmed stable during capture; input is disabled until a fresh inspect."
        self._snapshot_id = uuid.uuid4().hex[:10] if issue_action_token else ""
        self._snapshot_identity = after_identity if issue_action_token else None
        self._ocr_regions = []
        self._ocr_state = ""
        if self.ocr_only and self._include_ocr:
            try:
                with timed_stage("controller.ocr"):
                    self._ocr_regions = self._read_ocr_regions(image) if self._ocr_reader else []
                self._ocr_state = ScreenOCR.format(self._ocr_regions)
            except Exception as exc:
                error_lines = str(exc).strip().splitlines()
                error_detail = (error_lines[-1] if error_lines else "inference failed")[:180]
                print(
                    f"[OCR] Inference failed: {type(exc).__name__}: {error_detail}",
                    flush=True,
                )
                self._ocr_state = f"OCR unavailable: {type(exc).__name__}: {error_detail}"
            image_for_model = None
        elif self.ocr_only:
            self._ocr_state = "OCR and visual parsing were skipped by request; screenshot pixels are withheld by OCR-only mode."
            image_for_model = None
        else:
            image_for_model = image
            if self._include_ocr and self._ocr_reader is not None:
                try:
                    with timed_stage("controller.ocr"):
                        if hasattr(self._ocr_reader, "read"):
                            self._ocr_regions = self._read_ocr_regions(image)
                        elif hasattr(self._ocr_reader, "read_zoomed_band"):
                            header_regions = self._ocr_reader.read_zoomed_band(image, self._active_bounds)
                            panel_regions: list[OCRRegion] = []
                            if self._active_bounds:
                                left, top, right, bottom = self._active_bounds
                                panel_bounds = (
                                    left,
                                    min(bottom, top + 96),
                                    min(right, left + 320),
                                    min(bottom, top + 1000),
                                )
                                if panel_bounds[2] > panel_bounds[0] and panel_bounds[3] > panel_bounds[1]:
                                    panel_regions = self._ocr_reader.read_zoomed_band(
                                        image,
                                        panel_bounds,
                                        band_height=panel_bounds[3] - panel_bounds[1],
                                        tile_width=panel_bounds[2] - panel_bounds[0],
                                        tile_step=panel_bounds[2] - panel_bounds[0],
                                        scale=2,
                                    )
                            panel_regions = [
                                OCRRegion(f"P{index}", region.text, region.confidence,
                                          region.left, region.top, region.right, region.bottom)
                                for index, region in enumerate(panel_regions[:50], 1)
                            ]
                            self._ocr_regions = header_regions + panel_regions
                        else:
                            self._ocr_regions = []

                    if self._ocr_regions:
                        self._ocr_state = (
                            "Extracted screen text:\n"
                            + ScreenOCR.format(self._ocr_regions)
                        )
                except Exception as exc:
                    error_lines = str(exc).strip().splitlines()
                    error_detail = (error_lines[-1] if error_lines else "inference failed")[:180]
                    print(
                        f"[OCR] Inference failed: {type(exc).__name__}: {error_detail}",
                        flush=True,
                    )
                    self._ocr_state = f"Screen OCR unavailable: {type(exc).__name__}: {error_detail}"
        if self._include_ocr and self._ocr_reader is not None:
            diagnostics = getattr(self._ocr_reader, "last_read_diagnostics", {})
            if diagnostics.get("candidate_limit_hit"):
                detected = int(diagnostics.get("detected_candidates", 0))
                recognized = int(diagnostics.get("recognized_candidates", 0))
                self._ocr_state += (
                    f"\nOCR workload was bounded: recognized {recognized} of {detected} detected text candidates. "
                    "This screen read may omit lower-confidence text."
                )
        visual_details = ""
        if self._include_visual_grounding and self._visual_grounder is not None and not self.ocr_only:
            try:
                with timed_stage("controller.visual_grounder"):
                    annotated_image, visual_details = self._visual_grounder(image)
                    if annotated_image:
                        image_for_model = annotated_image
            except Exception as exc:
                visual_details = (
                    f"OmniParser grounding unavailable ({type(exc).__name__}: {str(exc)[:180]}). "
                    "Use the screenshot and available accessibility/browser data directly."
                )
        bounds_text = ""
        if self._active_bounds and not self.ocr_only:
            left, top, right, bottom = self._active_bounds
            bounds_text = (
                f" Active window bounds: x={left}..{right - 1}, y={top}..{bottom - 1}. "
                "Keep clicks inside this window; use focus_window before operating another app.\n"
            )
        coordinate_instructions = (
            "In OCR-only mode, describe the OCR text to click using target_text; Jev selects the text region. Do not use coordinates."
            if self.ocr_only and self._target_selector is not None
            else "In OCR-only mode, use OCR text as target_text. Coordinates are unavailable."
            if self.ocr_only
            else (
                "Click, drag waypoint, and drop coordinates use normalized values from 0 to 1000 across the screenshot width/height."
                if self.coordinate_mode == "normalized_1000"
                else "Click, drag waypoint, and drop coordinates use screenshot pixels from the top-left."
            )
        )
        token_text = (
            f"Snapshot ID: {self._snapshot_id}\n"
            if issue_action_token
            else "No action token was issued because focus was not confirmed stable. Inspect again before input.\n"
        )
        next_action_hint = (
            "For the next action, copy the Snapshot ID exactly. "
            if issue_action_token
            else "Call inspect to obtain an action-capable Snapshot ID before input. "
        )
        # Keep the original capture for controller-side OCR and visual-change
        # checks. The annotated image is the model-facing result only.
        self._last_screenshot = None if self.ocr_only else raw_screenshot
        self._has_captured_frame = True
        return ComputerControlResult(
            f"{prefix} Capture: scope={self._scope}, target={self._capture_target}, "
            f"backend={self._capture_backend}, image={self._width}x{self._height}, "
            f"origin=({self._origin_x},{self._origin_y}).\n"
            f"{token_text}"
            f"{bounds_text}"
            + next_action_hint + coordinate_instructions
            + (f"\n{self._ocr_state}" if self._ocr_state else "")
            + (f"\n{visual_details}" if visual_details else ""),
            image_for_model,
            status="ok" if issue_action_token else "uncertain",
            dispatched=False,
            snapshot_id=self._snapshot_id,
            ocr_regions=list(self._ocr_regions),
            raw_screenshot=None if self.ocr_only else raw_screenshot,
        )

    def _read_ocr_regions(self, image: bytes) -> list[OCRRegion]:
        """Read only the focused window when its bounds are known, then restore screen coordinates."""
        if self._ocr_reader is None:
            return []

        crop_origin = (0, 0)
        ocr_image = image
        if self._scope == "window" and self._active_bounds:
            try:
                from io import BytesIO
                from PIL import Image

                crop_started = time.perf_counter()
                with Image.open(BytesIO(image)) as source:
                    left, top, right, bottom = self._active_bounds
                    left = max(0, min(left, source.width))
                    top = max(0, min(top, source.height))
                    right = max(left, min(right, source.width))
                    bottom = max(top, min(bottom, source.height))
                    if right > left and bottom > top and (left, top, right, bottom) != (
                        0, 0, source.width, source.height
                    ):
                        encoded = BytesIO()
                        source.crop((left, top, right, bottom)).save(encoded, format="PNG")
                        ocr_image = encoded.getvalue()
                        crop_origin = (left, top)
                        log_duration(
                            "controller.ocr_window_crop",
                            crop_started,
                            source_width=source.width,
                            source_height=source.height,
                            crop_width=right - left,
                            crop_height=bottom - top,
                        )
            except Exception as exc:
                print(
                    f"[OCR] Window crop unavailable ({type(exc).__name__}); reading the captured image.",
                    flush=True,
                )

        regions = self._ocr_reader.read(ocr_image)
        dx, dy = crop_origin
        if not dx and not dy:
            return regions
        return [
            OCRRegion(
                region.ref,
                region.text,
                region.confidence,
                region.left + dx,
                region.top + dy,
                region.right + dx,
                region.bottom + dy,
            )
            for region in regions
        ]

    def _call(self, args: list[str], timeout: float = 5.0, **kwargs) -> subprocess.CompletedProcess:
        command = list(args)
        if self._env is not None:
            if not command:
                raise RuntimeError("Cannot execute an empty desktop action.")
            command[0] = self._command(command[0])
            # Scoped controllers must never inherit the daemon's host display
            # variables or PATH, even if an individual call supplied kwargs.
            kwargs["env"] = self._env
        result = self._runner(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        if result.returncode != 0:
            detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "input tool failed")
            raise RuntimeError(str(detail).strip()[:240])
        return result

    def _ensure_ydotoold(self) -> bool:
        if self._env is not None:
            # ydotoold emits global uinput events. A private DISPLAY or runtime
            # socket does not prove that its input device is isolated from the
            # host session, so scoped controllers must not reuse or start it.
            return False
        if self._ensure_wayland_daemon_override is not None:
            return self._ensure_wayland_daemon_override()
        socket_path = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / ".ydotool_socket"
        if socket_path.exists():
            return True
        # For ordinary daemon operation, keep the existing user-service path.
        if self._which("systemctl"):
            self._runner(["systemctl", "--user", "start", "ydotool.service"], capture_output=True, text=True, timeout=5)
            for _ in range(20):
                if socket_path.exists():
                    return True
                time.sleep(0.1)
        daemon = self._which("ydotoold")
        if daemon:
            try:
                subprocess.Popen(
                    [daemon], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True,
                )
                for _ in range(20):
                    if socket_path.exists():
                        return True
                    time.sleep(0.1)
            except OSError:
                pass
        return False

    def _require_wayland_uinput_scope(self) -> None:
        if self._env is not None:
            raise RuntimeError(
                "Wayland mouse input is unavailable for an explicitly scoped session: "
                "ydotool uses global uinput and input-device isolation was not established."
            )

    def _validate_snapshot(self, snapshot_id: str) -> None:
        if not self._snapshot_id:
            raise ValueError("Inspect the desktop before sending input.")
        if snapshot_id != self._snapshot_id:
            raise ValueError("This snapshot_id is not the controller's latest observation. Inspect again and use the returned ID.")
        current_identity, _ = self._read_active_window_state()
        if not self._snapshot_identity or current_identity != self._snapshot_identity:
            raise ValueError(
                "The focused window changed since that screenshot. Inspect the intended window again before input."
            )

    def _click_coordinates_to_pixels(self, x: int, y: int) -> tuple[int, int]:
        if self.coordinate_mode == "normalized_1000":
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise ValueError("Normalized click coordinates must be between 0 and 1000.")
            x = round(x * (self._width - 1) / 1000)
            y = round(y * (self._height - 1) / 1000)
        return x, y

    def _screenshot_to_desktop(self, x: int, y: int) -> tuple[int, int]:
        scale_x, scale_y = self._capture_scale
        return (
            round(x / scale_x) + self._origin_x,
            round(y / scale_y) + self._origin_y,
        )

    def _click(self, x: int, y: int, button: str) -> None:
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise ValueError(f"Click coordinates must be within the screenshot (0–{self._width - 1}, 0–{self._height - 1}).")
        if self._active_bounds:
            left, top, right, bottom = self._active_bounds
            if not (left <= x < right and top <= y < bottom):
                raise ValueError(
                    "Click rejected because it is outside the active window. "
                    "Focus the intended application first, then inspect the screen again."
                )
            if y <= top + round(36 * self._capture_scale[1]) and x >= right - round(40 * self._capture_scale[0]):
                raise ValueError(
                    "Click rejected because it targets the window-close corner. "
                    "Use close_application for an intentional window close; for visibility recovery, prefer hiding or moving an obstruction unless it is clearly disposable."
                )
        buttons = {"left": (1, "0xC0"), "right": (3, "0xC1"), "middle": (2, "0xC2")}
        if button not in buttons:
            raise ValueError("Mouse button must be left, right, or middle.")
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        self._position_cursor(x, y)
        if self.backend == "x11":
            self._call(["xdotool", "click", "--delay", "80", str(buttons[button][0])])
            return
        if self.backend == "wayland":
            self._call(["ydotool", "click", "--next-delay", "80", buttons[button][1]])
            return
        raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    def _position_cursor(self, x: int, y: int) -> None:
        desktop_x, desktop_y = self._screenshot_to_desktop(x, y)
        if self.backend == "x11":
            # --sync can hang against some fullscreen games/compositors. The
            # following click is sent by the same X server and remains ordered.
            self._call(["xdotool", "mousemove", str(desktop_x), str(desktop_y)])
            return
        if self.backend == "wayland":
            if self._env is not None:
                if self._wayland_window_adapter() != "hyprland":
                    self._require_wayland_uinput_scope()
                from src.tools.desktop import _hyprland_dispatch

                result = _hyprland_dispatch(
                    "movecursor", f"{desktop_x} {desktop_y}",
                    lua_expression=f"hl.dsp.cursor.move({{ x = {desktop_x}, y = {desktop_y} }})",
                    environ=self._env,
                )
                if result.returncode != 0 or "ok" not in result.stdout.lower():
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Hyprland rejected cursor positioning.")
                return
            if not self._ensure_ydotoold():
                raise RuntimeError("Wayland mouse control needs ydotoold and access to /dev/uinput.")
            env = os.environ if self._env is None else self._env
            if self._which("hyprctl") and env.get("HYPRLAND_INSTANCE_SIGNATURE"):
                self._move_wayland_cursor_to(desktop_x, desktop_y)
            else:
                self._call([
                    "ydotool", "mousemove", "--absolute",
                    str(desktop_x), str(desktop_y),
                ])
            return
        raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    def _move_wayland_cursor_to(self, end_x: int, end_y: int) -> None:
        """Move through uinput and verify the compositor cursor reaches the target.

        A compositor-native warp followed by a uinput click can leave the
        virtual input device at its previous location. Relative uinput motion
        keeps the pointer used for the click in sync with the visible cursor.
        """
        if self._env is not None:
            if self._wayland_window_adapter() != "hyprland":
                self._require_wayland_uinput_scope()
            from src.tools.desktop import _hyprland_dispatch

            result = _hyprland_dispatch(
                "movecursor", f"{end_x} {end_y}",
                lua_expression=f"hl.dsp.cursor.move({{ x = {end_x}, y = {end_y} }})",
                environ=self._env,
            )
            if result.returncode != 0 or "ok" not in result.stdout.lower():
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Hyprland rejected cursor positioning.")
            return
        if not self._ensure_ydotoold():
            raise RuntimeError("Wayland mouse control needs ydotoold and access to /dev/uinput.")

        def cursor() -> tuple[int, int]:
            data = json.loads(self._call(["hyprctl", "cursorpos", "-j"], timeout=2).stdout)
            return int(data["x"]), int(data["y"])

        start_x, start_y = cursor()
        delta_x, delta_y = end_x - start_x, end_y - start_y
        if not delta_x and not delta_y:
            return

        # Cursor positioning should be quick; feedback corrections below
        # compensate for the desktop's pointer acceleration.
        steps = 1
        moved_x = moved_y = 0
        for step in range(1, steps + 1):
            next_x = round(delta_x * step / steps)
            next_y = round(delta_y * step / steps)
            self._call([
                "ydotool", "mousemove", "--",
                str(next_x - moved_x), str(next_y - moved_y),
            ])
            moved_x, moved_y = next_x, next_y
            time.sleep(0.025)

        current = cursor()
        actual_x, actual_y = current[0] - start_x, current[1] - start_y
        ratio_x = actual_x / delta_x if delta_x and actual_x else 1.0
        ratio_y = actual_y / delta_y if delta_y and actual_y else 1.0
        for _ in range(3):
            error_x, error_y = end_x - current[0], end_y - current[1]
            if max(abs(error_x), abs(error_y)) <= 3:
                break
            correction_x = round(error_x / ratio_x) if abs(ratio_x) > 0.1 else error_x
            correction_y = round(error_y / ratio_y) if abs(ratio_y) > 0.1 else error_y
            if correction_x == 0 and error_x:
                correction_x = 1 if error_x > 0 else -1
            if correction_y == 0 and error_y:
                correction_y = 1 if error_y > 0 else -1
            self._call(["ydotool", "mousemove", "--", str(correction_x), str(correction_y)])
            time.sleep(0.06)
            current = cursor()
        if max(abs(end_x - current[0]), abs(end_y - current[1])) > 8:
            raise RuntimeError(
                f"Pointer reached ({current[0]}, {current[1]}), not requested position ({end_x}, {end_y})."
            )

    @property
    def drag_active(self) -> bool:
        return self._held_drag_button is not None

    def _begin_drag(self, x: int, y: int, button: str) -> None:
        if self._held_drag_button is not None:
            raise ValueError("A drag is already active; use drop before starting another drag.")
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise ValueError("Drag source must be inside the latest screenshot.")
        buttons = {"left": (1, "0x40"), "right": (3, "0x41"), "middle": (2, "0x42")}
        if button not in buttons:
            raise ValueError("Mouse button must be left, right, or middle.")
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        self._position_cursor(x, y)
        if self.backend == "x11":
            self._call(["xdotool", "mousedown", str(buttons[button][0])])
        elif self.backend == "wayland":
            if not self._ensure_ydotoold():
                raise RuntimeError("Wayland dragging needs ydotoold and access to /dev/uinput.")
            self._call(["ydotool", "click", "--next-delay", "60", buttons[button][1]])
        else:
            raise RuntimeError("No supported X11 or Wayland desktop session was detected.")
        self._held_drag_button = button
        self._held_drag_position = self._screenshot_to_desktop(x, y)

    def _move_held_drag(self, x: int, y: int) -> None:
        if self._held_drag_button is None:
            raise ValueError("No drag is active; call drag first.")
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise ValueError("Drag waypoint must be inside the latest screenshot.")
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        end_x, end_y = self._screenshot_to_desktop(x, y)
        if self.backend == "x11":
            start_x, start_y = self._held_drag_position or (end_x, end_y)
            steps = max(1, min(12, max(abs(end_x - start_x), abs(end_y - start_y)) // 24))
            for step in range(1, steps + 1):
                px = round(start_x + (end_x - start_x) * step / steps)
                py = round(start_y + (end_y - start_y) * step / steps)
                self._call(["xdotool", "mousemove", str(px), str(py)])
                time.sleep(0.025)
            self._held_drag_position = (end_x, end_y)
            return
        if self.backend != "wayland":
            raise RuntimeError("No supported X11 or Wayland desktop session was detected.")
        if not self._ensure_ydotoold():
            raise RuntimeError("Wayland dragging needs ydotoold and access to /dev/uinput.")
        env = os.environ if self._env is None else self._env
        if not (self._which("hyprctl") and env.get("HYPRLAND_INSTANCE_SIGNATURE")):
            self._call(["ydotool", "mousemove", "--absolute", str(end_x), str(end_y)])
            time.sleep(0.12)
            return

        def cursor() -> tuple[int, int]:
            data = json.loads(self._call(["hyprctl", "cursorpos", "-j"], timeout=2).stdout)
            return int(data["x"]), int(data["y"])

        start_x, start_y = cursor()
        delta_x, delta_y = end_x - start_x, end_y - start_y
        steps = max(1, min(16, max(abs(delta_x), abs(delta_y)) // 24))
        moved_x = moved_y = 0
        for step in range(1, steps + 1):
            next_x = round(delta_x * step / steps)
            next_y = round(delta_y * step / steps)
            self._call([
                "ydotool", "mousemove", "--",
                str(next_x - moved_x), str(next_y - moved_y),
            ])
            moved_x, moved_y = next_x, next_y
            time.sleep(0.025)
        current = cursor()
        actual_x, actual_y = current[0] - start_x, current[1] - start_y
        ratio_x = actual_x / delta_x if delta_x and actual_x else 1.0
        ratio_y = actual_y / delta_y if delta_y and actual_y else 1.0
        for _ in range(3):
            error_x, error_y = end_x - current[0], end_y - current[1]
            if max(abs(error_x), abs(error_y)) <= 3:
                break
            correction_x = round(error_x / ratio_x) if abs(ratio_x) > 0.1 else error_x
            correction_y = round(error_y / ratio_y) if abs(ratio_y) > 0.1 else error_y
            if correction_x == 0 and error_x:
                correction_x = 1 if error_x > 0 else -1
            if correction_y == 0 and error_y:
                correction_y = 1 if error_y > 0 else -1
            self._call(["ydotool", "mousemove", "--", str(correction_x), str(correction_y)])
            time.sleep(0.06)
            current = cursor()
        if max(abs(end_x - current[0]), abs(end_y - current[1])) > 8:
            raise RuntimeError(
                f"Drag reached ({current[0]}, {current[1]}), not requested waypoint ({end_x}, {end_y})."
            )

    def _release_drag(self) -> None:
        button = self._held_drag_button
        if button is None:
            raise ValueError("No drag is active; nothing to drop.")
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        buttons = {"left": (1, "0x80"), "right": (3, "0x81"), "middle": (2, "0x82")}
        try:
            if self.backend == "x11":
                self._call(["xdotool", "mouseup", str(buttons[button][0])])
            elif self.backend == "wayland":
                self._call(["ydotool", "click", "--next-delay", "60", buttons[button][1]])
            else:
                raise RuntimeError("No supported X11 or Wayland desktop session was detected.")
        finally:
            self._held_drag_button = None
            self._held_drag_position = None

    def release_held_drag(self) -> None:
        """Release any mouse button left down by a multi-step drag."""
        with self._action_lock:
            if self._held_drag_button is not None:
                self._release_drag()

    def _drag(
        self, x: int, y: int, end_x: int, end_y: int, button: str = "left", *,
        restrict_to_active_window: bool = False,
    ) -> None:
        """Drag from one screenshot point to another; useful for window placement on any WM."""
        points = ((x, y, "start"), (end_x, end_y, "end"))
        for px, py, label in points:
            if not (0 <= px < self._width and 0 <= py < self._height):
                raise ValueError(f"Drag {label} must be inside the screenshot (0–{self._width - 1}, 0–{self._height - 1}).")
        if restrict_to_active_window and self._active_bounds:
            left, top, right, bottom = self._active_bounds
            if not (left <= x < right and top <= y < bottom):
                raise ValueError("Drag must begin inside the active window. Inspect it and start on the control or title bar to move.")
            if y <= top + round(36 * self._capture_scale[1]) and x >= right - round(40 * self._capture_scale[0]):
                raise ValueError("Drag start is on the window-close corner; choose a title-bar point away from window controls.")
        buttons = {"left": (1, "0x40", "0x80"), "right": (3, "0x41", "0x81"), "middle": (2, "0x42", "0x82")}
        if button not in buttons:
            raise ValueError("Mouse button must be left, right, or middle.")
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        sx, sy = self._screenshot_to_desktop(x, y)
        ex, ey = self._screenshot_to_desktop(end_x, end_y)
        if self.backend == "x11":
            self._call(["xdotool", "mousemove", str(sx), str(sy)])
            self._call(["xdotool", "mousedown", str(buttons[button][0])])
            try:
                steps = max(1, min(16, max(abs(ex - sx), abs(ey - sy)) // 24))
                for step in range(1, steps + 1):
                    px = round(sx + (ex - sx) * step / steps)
                    py = round(sy + (ey - sy) * step / steps)
                    self._call(["xdotool", "mousemove", str(px), str(py)])
                    time.sleep(0.025)
            finally:
                self._call(["xdotool", "mouseup", str(buttons[button][0])])
            return
        if self.backend == "wayland":
            if not self._ensure_ydotoold():
                raise RuntimeError("Wayland dragging needs ydotoold and access to /dev/uinput.")
            env = os.environ if self._env is None else self._env
            if self._which("hyprctl") and env.get("HYPRLAND_INSTANCE_SIGNATURE"):
                from src.tools.desktop import _hyprland_dispatch

                # Hyprland gives exact screenshot-to-cursor positioning. Use
                # relative ydotool motion after pressing so the compositor sees
                # real pointer-move events. ydotool absolute values use a device
                # range rather than screenshot pixels, so they cannot safely be
                # passed screen coordinates here.
                result = _hyprland_dispatch(
                    "movecursor", f"{sx} {sy}",
                    lua_expression=f"hl.dsp.cursor.move({{ x = {sx}, y = {sy} }})",
                    environ=self._env,
                )
                if result.returncode != 0 or "ok" not in result.stdout.lower():
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Hyprland rejected drag positioning.")
                def hyprland_cursor() -> tuple[int, int]:
                    data = json.loads(self._call(["hyprctl", "cursorpos", "-j"], timeout=2).stdout)
                    return int(data["x"]), int(data["y"])

                drag_origin = hyprland_cursor()
                self._call(["ydotool", "click", "--next-delay", "60", buttons[button][1]])
                try:
                    delta_x, delta_y = ex - sx, ey - sy
                    steps = max(1, min(12, max(abs(delta_x), abs(delta_y)) // 24))
                    moved_x = moved_y = 0
                    for step in range(1, steps + 1):
                        next_x = round(delta_x * step / steps)
                        next_y = round(delta_y * step / steps)
                        self._call([
                            "ydotool", "mousemove", "--",
                            str(next_x - moved_x), str(next_y - moved_y),
                        ])
                        moved_x, moved_y = next_x, next_y
                        time.sleep(0.025)
                    # uinput relative deltas are subject to the compositor's
                    # pointer acceleration. Measure the actual cursor travel and
                    # correct it while the button is still held, keeping the
                    # requested drag endpoint in screenshot pixels.
                    current = hyprland_cursor()
                    actual_x, actual_y = current[0] - drag_origin[0], current[1] - drag_origin[1]
                    ratio_x = actual_x / delta_x if delta_x and actual_x else 1.0
                    ratio_y = actual_y / delta_y if delta_y and actual_y else 1.0
                    for _ in range(3):
                        error_x, error_y = ex - current[0], ey - current[1]
                        if max(abs(error_x), abs(error_y)) <= 3:
                            break
                        correction_x = round(error_x / ratio_x) if abs(ratio_x) > 0.1 else error_x
                        correction_y = round(error_y / ratio_y) if abs(ratio_y) > 0.1 else error_y
                        if correction_x == 0 and error_x:
                            correction_x = 1 if error_x > 0 else -1
                        if correction_y == 0 and error_y:
                            correction_y = 1 if error_y > 0 else -1
                        self._call([
                            "ydotool", "mousemove", "--",
                            str(correction_x), str(correction_y),
                        ])
                        time.sleep(0.06)
                        current = hyprland_cursor()
                    if max(abs(ex - current[0]), abs(ey - current[1])) > 8:
                        raise RuntimeError(
                            f"Hyprland drag ended at ({current[0]}, {current[1]}), "
                            f"short of requested release point ({ex}, {ey})."
                        )
                finally:
                    self._call(["ydotool", "click", "--next-delay", "60", buttons[button][2]])
                return
            self._call(["ydotool", "mousemove", "--absolute", str(sx), str(sy)])
            self._call(["ydotool", "click", "--next-delay", "60", buttons[button][1]])
            try:
                self._call(["ydotool", "mousemove", "--absolute", str(ex), str(ey)])
                time.sleep(0.35)
            finally:
                self._call(["ydotool", "click", "--next-delay", "60", buttons[button][2]])
            return
        raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    def _window_move_modifier(self) -> str:
        """Infer the compositor's configured drag-to-move modifier where possible."""
        from src.tools.desktop import get_active_backend

        backend = get_active_backend(environ=self._env).name
        candidates = {
            "hyprland": [Path.home() / ".config/hypr/hyprland.lua", Path.home() / ".config/hypr/hyprland.conf"],
            "sway": [Path.home() / ".config/sway/config"],
            "i3": [Path.home() / ".config/i3/config"],
        }.get(backend, [])
        for path in candidates:
            try:
                config_text = path.read_text(encoding="utf-8", errors="ignore")
                pattern = r'(?:local\s+)?\$?mainMod\s*=\s*["\']?([A-Za-z0-9_]+)'
                if backend in {"sway", "i3"}:
                    pattern = r"^\s*set\s+\$mod\s+([A-Za-z0-9_]+)"
                match = re.search(pattern, config_text, re.MULTILINE | re.IGNORECASE)
                if match:
                    value = match.group(1).casefold()
                    if value in {"alt", "mod1"}:
                        return "alt"
                    if value in {"super", "meta", "mod4", "win", "windows"}:
                        return "super"
            except OSError:
                continue
        if backend in {"gnome", "generic_desktop"} and self.backend == "x11":
            return "alt"
        return "super"

    def _type(self, text: str) -> None:
        if len(text) > self.max_text_length:
            raise ValueError(f"Text is too long; the limit is {self.max_text_length} characters per action.")
        if self.backend == "x11":
            self._call(["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text], timeout=15)
        elif self.backend == "wayland":
            if self._which("wtype"):
                self._call(["wtype", "--", text], timeout=15)
            elif self._ensure_ydotoold():
                self._call(["ydotool", "type", "--", text], timeout=15)
            else:
                raise RuntimeError("Wayland text input needs wtype or a running ydotoold daemon.")
        else:
            raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    @staticmethod
    def _parse_key(value: str) -> tuple[str, list[str]]:
        key = "+".join(part.strip().lower() for part in (value or "").split("+") if part.strip())
        aliases = {"return": "enter", "esc": "escape", "pgup": "pageup", "pgdn": "pagedown"}
        key = "+".join(aliases.get(part, part) for part in key.split("+"))
        if key == "ctrl+plus":
            return "equal", ["ctrl", "shift"]
        if key in _ALLOWED_COMBOS:
            parts = key.split("+")
            return parts[-1], parts[:-1]
        if "+" in key:
            raise ValueError("That key combination is not allowed. Use a listed navigation key or common Ctrl shortcut.")
        if key not in _X11_KEYS and key not in _LETTER_KEYCODES:
            raise ValueError("Supported keys: Enter, Tab, Escape, Backspace, Delete, arrows, Home, End, PageUp, PageDown, and Space.")
        return key, []

    def _press(self, value: str) -> None:
        key, modifiers = self._parse_key(value)
        if self.backend == "x11":
            key_name = _X11_KEYS.get(key, key)
            combo = "+".join([*modifiers, key_name])
            self._call(["xdotool", "key", "--clearmodifiers", combo])
            return
        if self.backend == "wayland":
            env = os.environ if self._env is None else self._env
            if modifiers and self._which("hyprctl") and env.get("HYPRLAND_INSTANCE_SIGNATURE"):
                from src.tools.desktop import _hyprland_dispatch

                modifier = "+".join(item.upper() for item in modifiers)
                key_name = _WAYLAND_KEYS.get(key, key).upper()
                result = _hyprland_dispatch(
                    "sendshortcut",
                    f"{modifier},{key_name},",
                    lua_expression=(
                        "hl.dsp.send_shortcut({ "
                        f'mods = "{modifier}", key = "{key_name}"'
                        " })"
                    ),
                    environ=self._env,
                )
                if result.returncode != 0 or "ok" not in result.stdout.lower():
                    raise RuntimeError(
                        result.stderr.strip() or result.stdout.strip() or "Hyprland rejected the keyboard shortcut."
                    )
                return
            if self._which("wtype"):
                prefix = []
                for modifier in modifiers:
                    prefix.extend(["-M", modifier])
                # Prefer wtype's positional text path for letter shortcuts; some
                # wtype/compositor combinations drop modifiers through the -k path.
                if modifiers and key in _LETTER_KEYCODES:
                    prefix.append(key)
                else:
                    prefix.extend(["-k", _WAYLAND_KEYS.get(key, key)])
                for modifier in reversed(modifiers):
                    prefix.extend(["-m", modifier])
                self._call(["wtype", *prefix])
                return
            if self._ensure_ydotoold():
                sequence: list[str] = []
                for modifier in modifiers:
                    sequence.append(f"{_MODIFIER_CODES[modifier]}:1")
                keycode = _YDOTOOL_KEYCODES.get(key) or _LETTER_KEYCODES.get(key)
                if keycode is None:
                    raise ValueError("That key is not supported by ydotool fallback.")
                sequence.extend([f"{keycode}:1", f"{keycode}:0"])
                for modifier in reversed(modifiers):
                    sequence.append(f"{_MODIFIER_CODES[modifier]}:0")
                self._call(["ydotool", "key", *sequence])
                return
        raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    def _scroll(self, direction: str, amount: int) -> None:
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("Scroll direction must be up, down, left, or right.")
        amount = min(max(int(amount), 1), 8)
        if self.backend == "wayland":
            self._require_wayland_uinput_scope()
        if self.backend == "x11":
            button = {"up": 4, "down": 5, "left": 6, "right": 7}[direction]
            self._call(["xdotool", "click", "--repeat", str(amount), str(button)])
        elif self.backend == "wayland":
            if not self._ensure_ydotoold():
                raise RuntimeError("Wayland scrolling needs ydotoold and access to /dev/uinput.")
            # In Linux uinput / ydotool mousemove -w:
            # Vertical wheel: positive dy = up, negative dy = down.
            # Horizontal wheel: positive dx = right, negative dx = left.
            dx, dy = 0, 0
            if direction == "up":
                dy = amount
            elif direction == "down":
                dy = -amount
            elif direction == "left":
                dx = -amount
            elif direction == "right":
                dx = amount
            self._call(["ydotool", "mousemove", "-w", "--", str(dx), str(dy)])
        else:
            raise RuntimeError("No supported X11 or Wayland desktop session was detected.")

    @_serialized
    def run_sequence(
        self,
        snapshot_id: str,
        actions: list[dict],
        *,
        screenshot_delay_seconds: float | None = None,
        include_ocr: bool | None = None,
        include_visual_grounding: bool | None = None,
        goal: str = "",
        expected_application: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ComputerControlResult:
        """Execute a model-selected sequence while refusing to reuse old geometry."""
        if not isinstance(actions, list) or not actions:
            return ComputerControlResult("Sequence needs at least one action.", status="invalid_input")
        if len(actions) > self.max_sequence_actions:
            return ComputerControlResult(
                f"Sequence has {len(actions)} actions; the current per-call limit is {self.max_sequence_actions}.",
                status="invalid_input",
            )
        sequence_text_length = 0
        for index, step in enumerate(actions):
            if not isinstance(step, dict):
                return ComputerControlResult(
                    f"Sequence action {index + 1} must be an object.", status="invalid_input"
                )
            if str(step.get("action", "")).strip().lower() != "type":
                continue
            text = step.get("text", "")
            if not isinstance(text, str):
                return ComputerControlResult(
                    f"Sequence action {index + 1} text must be a string.", status="invalid_input"
                )
            if len(text) > self.max_text_length:
                return ComputerControlResult(
                    f"Sequence action {index + 1} exceeds the per-action text limit of "
                    f"{self.max_text_length} characters.", status="invalid_input"
                )
            sequence_text_length += len(text)
            if sequence_text_length > self.max_sequence_text_length:
                return ComputerControlResult(
                    f"Sequence text totals {sequence_text_length} characters; the per-sequence limit is "
                    f"{self.max_sequence_text_length}. No actions were run.",
                    status="invalid_input",
                )
        started = time.monotonic()
        current_snapshot = snapshot_id
        results: list[str] = []
        latest: ComputerControlResult | None = None
        any_dispatched: bool | None = False
        last_concrete_action = ""
        for index, step in enumerate(actions):
            if cancel_event and cancel_event.is_set():
                return ComputerControlResult(
                    "Sequence cancelled at an action boundary.\n" + "\n".join(results),
                    latest.screenshot if latest else None,
                    status="cancelled",
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id if latest else "",
                    raw_screenshot=latest.raw_screenshot if latest else None,
                )
            if time.monotonic() - started >= self.sequence_timeout_seconds:
                return ComputerControlResult(
                    f"Sequence time budget reached at an action boundary after {index} action(s); "
                    "the active action was allowed to finish. Inspect the current state before continuing.\n"
                    + "\n".join(results),
                    latest.screenshot if latest else None,
                    status="timed_out",
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id if latest else "",
                    raw_screenshot=latest.raw_screenshot if latest else None,
                )
            if not isinstance(step, dict):
                return ComputerControlResult(
                    f"Sequence action {index + 1} must be an object.\n" + "\n".join(results),
                    latest.screenshot if latest else None,
                    status="invalid_input",
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id if latest else "",
                    raw_screenshot=latest.raw_screenshot if latest else None,
                )
            action = str(step.get("action", "")).strip().lower()
            if action not in {"click", "drag", "type", "press", "scroll", "wait"}:
                return ComputerControlResult(
                    f"Sequence action {index + 1} has unsupported operation {action!r}.\n" + "\n".join(results),
                    latest.screenshot if latest else None,
                    status="invalid_input",
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id if latest else "",
                    raw_screenshot=latest.raw_screenshot if latest else None,
                )
            # Keyboard/text input and pauses have no reusable screen coordinates.
            # Allow click followed by typing into the selected control, keypresses
            # such as Enter to submit, and wait/delay pauses between or after.
            key_name = str(step.get("key", "")).strip().casefold().replace(" ", "")
            move_to_text_start_after_ocr_target = (
                last_concrete_action == "ocr_click"
                and action == "press"
                and key_name == "home"
            )
            select_text_to_line_end = (
                last_concrete_action == "text_start"
                and action == "press"
                and key_name == "shift+end"
            )
            coordinate_free_continuation = (
                action == "wait"
            ) or (
                last_concrete_action in {"click", "ocr_click"}
                and action == "type"
            ) or (
                last_concrete_action == "text_selected"
                and action == "type"
            ) or (
                move_to_text_start_after_ocr_target
                or select_text_to_line_end
            ) or (
                last_concrete_action == "type"
                and action == "press"
            ) or (
                # A text label is resolved against current OCR for each step;
                # it does not reuse the prior click's pixel coordinates.
                last_concrete_action in {"click", "ocr_click", "type"}
                and action == "click"
                and bool(str(step.get("target_text", "")).strip())
                and bool(include_ocr or self._include_ocr)
            )
            if index and not coordinate_free_continuation:
                pause_reason = (
                    "Ctrl+A was not sent because Linux single-line fields may move the caret instead of selecting text. "
                    "For replacement, click the OCR-recognized current value, press Home, press Shift+End, then type the requested value."
                    if last_concrete_action == "ocr_click"
                    and action == "press"
                    and key_name == "ctrl+a"
                    else "Sequence paused before this input because continuing could reuse a target or assume focus."
                )
                return ComputerControlResult(
                    f"{pause_reason} Adam must choose the next action from the fresh observation.\n"
                    + "\n".join(results) + "\n" + (latest.message if latest else ""),
                    latest.screenshot if latest else None,
                    status="partial",
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id if latest else "",
                    raw_screenshot=latest.raw_screenshot if latest else None,
                )
            if action == "wait":
                delay_sec = min(max(float(step.get("seconds") or step.get("amount") or 1.0), 0.0), 10.0)
                time.sleep(delay_sec)
                results.append(f"Step {index + 1}/{len(actions)} (wait): ok; Waited {delay_sec:g}s.")
                if index == len(actions) - 1:
                    latest = self.run(
                        action="inspect",
                        snapshot_id="",
                        screenshot_delay_seconds=0,
                        goal=goal,
                        expected_application=expected_application,
                    )
                continue
            last_concrete_action = (
                "text_start" if move_to_text_start_after_ocr_target else
                "text_selected" if select_text_to_line_end else
                "ocr_click"
                if action == "click"
                and bool(str(step.get("target_text", "")).strip())
                and bool(include_ocr or self._include_ocr)
                else action
            )
            kwargs = {key: value for key, value in step.items() if key != "action"}
            if include_ocr is not None:
                kwargs.setdefault("include_ocr", include_ocr)
            if include_visual_grounding is not None:
                kwargs.setdefault("include_visual_grounding", include_visual_grounding)
            latest = self.run(
                action=action,
                snapshot_id=current_snapshot,
                screenshot_delay_seconds=screenshot_delay_seconds,
                goal=goal,
                expected_application=expected_application,
                **kwargs,
            )
            if latest.dispatched is True:
                any_dispatched = True
            elif latest.dispatched is None and any_dispatched is False:
                any_dispatched = None
            results.append(f"Step {index + 1}/{len(actions)} ({action}): {latest.status}; {latest.message}")
            if latest.status != "ok":
                return ComputerControlResult(
                    "Sequence stopped after a step did not complete successfully.\n" + "\n".join(results),
                    latest.screenshot,
                    status=latest.status,
                    dispatched=any_dispatched,
                    snapshot_id=latest.snapshot_id,
                    raw_screenshot=latest.raw_screenshot,
                )
            if not latest.snapshot_id:
                return ComputerControlResult(
                    "Sequence stopped because the latest action did not produce a usable fresh snapshot.\n"
                    + "\n".join(results),
                    latest.screenshot,
                    status="partial",
                    dispatched=True,
                    snapshot_id=latest.snapshot_id,
                    raw_screenshot=latest.raw_screenshot,
                )
            current_snapshot = latest.snapshot_id
        return ComputerControlResult(
            f"Sequence executed {len(actions)} action(s). This reports dispatch and observations, not task completion.\n"
            + "\n".join(results) + "\n" + (latest.message if latest else ""),
            latest.screenshot if latest else None,
            status="ok",
            dispatched=any_dispatched,
            snapshot_id=latest.snapshot_id if latest else "",
            raw_screenshot=latest.raw_screenshot if latest else None,
        )

    @_serialized
    def run(
        self,
        action: str,
        snapshot_id: str = "",
        x: int | None = None,
        y: int | None = None,
        end_x: int | None = None,
        end_y: int | None = None,
        waypoints: list[dict[str, int]] | None = None,
        button: str = "left",
        modifier: str = "none",
        text: str = "",
        key: str = "",
        direction: str = "down",
        amount: int = 3,
        scope: str | None = None,
        screenshot_delay_seconds: float | None = None,
        target_text: str = "",
        ocr_region_ref: str = "",
        include_ocr: bool | None = None,
        include_visual_grounding: bool | None = None,
        goal: str = "",
        expected_application: str | None = None,
        readiness_timeout_seconds: float | None = 15.0,
    ) -> ComputerControlResult:
        if not self.enabled:
            return ComputerControlResult(
                "Computer control is disabled in config.yaml.", status="unavailable"
            )
        action = (action or "inspect").strip().lower()
        previous_screenshot = self._last_screenshot
        previous_ocr_text = ScreenOCR.text_signature(self._ocr_regions)
        if action == "inspect":
            self._include_ocr = (self.ocr_only if include_ocr is None else bool(include_ocr))
            self._include_visual_grounding = bool(include_visual_grounding)
        elif include_ocr is not None:
            self._include_ocr = bool(include_ocr)
            if include_visual_grounding is not None:
                self._include_visual_grounding = bool(include_visual_grounding)
        with timed_stage("controller.screenshot_delay_selection"):
            max_configured_delay = screenshot_delay_for_focused_window(
                self.screenshot_delay_seconds,
                self.browser_screenshot_delay_seconds,
                expected_application,
                environ=self._env,
            )
            if screenshot_delay_seconds is None:
                # The first observation reads the currently focused app, so it
                # can use the normal short settle. Later browser inspections and
                # identified app transitions retain the longer wait in case a
                # page is still loading after launch or navigation.
                delay = (
                    max_configured_delay
                    if action == "inspect" and (expected_application or self._has_captured_frame)
                    else self.screenshot_delay_seconds
                )
            else:
                try:
                    delay = float(screenshot_delay_seconds)
                except (TypeError, ValueError):
                    delay = max_configured_delay
                if not math.isfinite(delay):
                    delay = max_configured_delay
                # Agent supplied waits may shorten the configured app-aware
                # settle time, but cannot override it with seconds of avoidable
                # latency. Longer waits belong in configuration.
                delay = min(max(delay, 0.0), max_configured_delay)
        if action == "inspect":
            try:
                requested_scope = (scope or "window").strip().lower()
                if requested_scope not in {"monitor", "window", "desktop"}:
                    raise ValueError("Screenshot scope must be 'window', 'monitor', or 'desktop'.")
                self._scope = requested_scope
                label = {
                    "window": "focused application",
                    "monitor": "active monitor",
                    "desktop": "full desktop",
                }[self._scope]
                if delay:
                    with timed_stage("controller.screenshot_settle", configured_delay_ms=round(delay * 1000)):
                        time.sleep(delay)
                return self._capture(
                    f"Inspected the {label} after waiting {delay:g}s.",
                    expected_application=expected_application,
                    readiness_timeout_seconds=readiness_timeout_seconds,
                    # An explicit current-screen read has no app transition to
                    # wait for. Launch/focus follow-ups pass an expected app;
                    # post-action captures still use the normal readiness poll.
                    wait_until_ready=bool(expected_application),
                )
            except Exception as exc:
                self._snapshot_id = ""
                return ComputerControlResult(
                    f"Could not capture the desktop screenshot: {type(exc).__name__}: {str(exc)[:240]}",
                    status="failed",
                )
        message = ""
        action_succeeded = False
        action_attempted = False
        failure_status = "failed"
        try:
            if not self.available:
                raise RuntimeError(f"Computer input is unavailable for the detected {self.backend} session.")
            self._validate_snapshot(snapshot_id)
            if action == "click":
                internal_ocr_target = False
                if self.ocr_only and ocr_region_ref.strip():
                    region = next(
                        (item for item in self._ocr_regions if item.ref.casefold() == ocr_region_ref.strip().casefold()),
                        None,
                    )
                    if region is None:
                        raise ValueError("That OCR reference is not part of the current snapshot.")
                    x, y = region.center
                    internal_ocr_target = True
                    selection_message = f"Clicked Jev-selected OCR target {region.ref}."
                elif target_text.strip() and not self.ocr_only:
                    if (
                        not self._ocr_regions
                        and self._include_ocr
                        and self._ocr_reader is not None
                        and self._last_screenshot
                    ):
                        try:
                            with timed_stage("controller.ocr_target_lookup"):
                                self._ocr_regions = self._read_ocr_regions(self._last_screenshot)
                            self._ocr_state = ScreenOCR.format(self._ocr_regions)
                        except Exception as exc:
                            selection_message = (
                                f"OCR target lookup failed ({type(exc).__name__}: {str(exc)[:160]})."
                            )
                    region, selection_message = _match_ocr_region(target_text, self._ocr_regions)
                    if region is None:
                        raise ValueError(selection_message)
                    x, y = region.center
                    internal_ocr_target = True
                elif self.ocr_only and self._target_selector is not None:
                    if not target_text.strip():
                        raise ValueError("OCR target selection requires target_text; coordinates are not accepted in this mode.")
                    region, selection_message = self._target_selector(
                        goal, target_text, self._ocr_state, self._ocr_regions
                    )
                    if region is None:
                        raise ValueError(selection_message)
                    x, y = region.center
                    internal_ocr_target = True
                elif self.ocr_only and target_text.strip():
                    raise ValueError("Jev target selection is disabled; no OCR target was clicked.")
                if x is None or y is None:
                    raise ValueError("Click requires x/y coordinates or a text target from the latest OCR snapshot.")
                requested_x, requested_y = int(x), int(y)
                pixel_x, pixel_y = (
                    (requested_x, requested_y)
                    if internal_ocr_target
                    else self._click_coordinates_to_pixels(requested_x, requested_y)
                )
                action_attempted = True
                self._click(pixel_x, pixel_y, button.lower())
                if self.coordinate_mode == "normalized_1000":
                    message = (
                        f"Clicked {button} at normalized ({requested_x}, {requested_y}) "
                        f"→ screenshot pixels ({pixel_x}, {pixel_y})."
                    )
                else:
                    message = f"Clicked {button} at ({pixel_x}, {pixel_y})."
                if target_text.strip() and (self.ocr_only or internal_ocr_target):
                    message = f"{selection_message} Clicked the selected OCR region."
                elif self.ocr_only and ocr_region_ref.strip():
                    message = f"{selection_message}"
            elif action == "drag_path":
                if x is None or y is None or not waypoints:
                    raise ValueError("Drag requires a source x/y and at least one waypoint.")
                source_x, source_y = self._click_coordinates_to_pixels(int(x), int(y))
                pixel_waypoints = []
                for index, point in enumerate(waypoints):
                    if not isinstance(point, dict) or point.get("x") is None or point.get("y") is None:
                        raise ValueError(f"Drag waypoint {index + 1} requires x and y coordinates.")
                    pixel_waypoints.append(
                        self._click_coordinates_to_pixels(int(point["x"]), int(point["y"]))
                    )
                action_attempted = True
                self._begin_drag(source_x, source_y, button.lower())
                try:
                    for waypoint_x, waypoint_y in pixel_waypoints:
                        self._move_held_drag(waypoint_x, waypoint_y)
                except Exception:
                    try:
                        self._release_drag()
                    except Exception:
                        pass
                    raise
                action_succeeded = True
                final_x, final_y = pixel_waypoints[-1]
                message = (
                    f"Drag is active with {button} held. Followed {len(pixel_waypoints)} waypoint(s); "
                    f"cursor is at ({final_x}, {final_y}). Call drop to release it."
                )
            elif action == "drop":
                if self._held_drag_button is None:
                    raise ValueError("No drag is active; call drag first.")
                action_attempted = True
                try:
                    if (x is None) != (y is None):
                        raise ValueError("Drop destination requires both x and y, or neither.")
                    if x is not None and y is not None:
                        drop_x, drop_y = self._click_coordinates_to_pixels(int(x), int(y))
                        self._move_held_drag(drop_x, drop_y)
                    self._release_drag()
                except Exception:
                    if self._held_drag_button is not None:
                        try:
                            self._release_drag()
                        except Exception:
                            pass
                    raise
                action_succeeded = True
                message = "Released the held mouse button at the requested drop point."
            elif action == "drag":
                if x is None or y is None or end_x is None or end_y is None:
                    raise ValueError("Drag requires x/y and end_x/end_y coordinates from the current screenshot.")
                start_x, start_y = self._click_coordinates_to_pixels(int(x), int(y))
                finish_x, finish_y = self._click_coordinates_to_pixels(int(end_x), int(end_y))
                drag_modifier = modifier.strip().lower()
                if drag_modifier == "window":
                    drag_modifier = self._window_move_modifier()
                if drag_modifier not in {"none", "alt", "super"}:
                    raise ValueError("Drag modifier must be none, alt, super, or window (detect the WM's move modifier).")
                action_attempted = True
                if drag_modifier == "none":
                    self._drag(start_x, start_y, finish_x, finish_y, button.lower())
                else:
                    key_name = "Alt_L" if drag_modifier == "alt" else "Super_L"
                    if self.backend == "x11":
                        self._call(["xdotool", "keydown", key_name])
                        try:
                            self._drag(
                                start_x, start_y, finish_x, finish_y, button.lower(),
                                restrict_to_active_window=True,
                            )
                        finally:
                            self._call(["xdotool", "keyup", key_name])
                    elif self.backend == "wayland":
                        if not self._ensure_ydotoold():
                            raise RuntimeError("Wayland window dragging needs ydotoold and access to /dev/uinput.")
                        self._call(["ydotool", "key", f"{_MODIFIER_CODES[drag_modifier]}:1"])
                        try:
                            self._drag(
                                start_x, start_y, finish_x, finish_y, button.lower(),
                                restrict_to_active_window=True,
                            )
                        finally:
                            self._call(["ydotool", "key", f"{_MODIFIER_CODES[drag_modifier]}:0"])
                message = f"Dragged {button} from ({start_x}, {start_y}) to ({finish_x}, {finish_y}) with {drag_modifier} modifier."
            elif action == "type":
                action_attempted = True
                self._type(text)
                message = f"Typed {len(text)} characters into the currently focused control."
            elif action == "press":
                action_attempted = True
                self._press(key)
                message = f"Pressed {key}."
            elif action == "scroll":
                action_attempted = True
                if x is not None and y is not None:
                    pixel_x, pixel_y = self._click_coordinates_to_pixels(int(x), int(y))
                    self._position_cursor(pixel_x, pixel_y)
                self._scroll(direction.lower(), amount)
                message = f"Scrolled {direction} by {min(max(int(amount), 1), 8)} steps."
            else:
                raise ValueError("Action must be inspect, click, drag, type, press, or scroll.")
            action_succeeded = True
        except ValueError as exc:
            failure_status = "invalid_input"
            message = f"Action stopped: {exc}"
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            failure_status = "uncertain" if action_attempted else "failed"
            message = f"Action stopped: {exc}"
        if delay:
            time.sleep(delay)
        try:
            captured = self._capture(
                f"{message or 'Action dispatched.'} Fresh screenshot captured after {delay:g}s.",
                issue_action_token=action_succeeded,
                expected_application=expected_application,
                readiness_timeout_seconds=readiness_timeout_seconds,
            )
            captured.status = (
                "ok" if action_succeeded and captured.snapshot_id
                else "uncertain" if action_succeeded
                else failure_status
            )
            captured.dispatched = action_succeeded if action_succeeded else (None if action_attempted else False)
            if (
                action == "click"
                and action_succeeded
                and previous_screenshot
                and self._last_screenshot
                and _screen_state_unchanged(
                    previous_screenshot,
                    self._last_screenshot,
                    previous_ocr_text,
                    ScreenOCR.text_signature(captured.ocr_regions),
                )
            ):
                captured.message += (
                    " The fresh screen appears unchanged after this click. The input was dispatched, "
                    "but its effect is not visually confirmed; reassess the current target before claiming completion."
                )
            return captured
        except Exception as exc:
            self._snapshot_id = ""
            return ComputerControlResult(
                f"{message or 'The action may have been dispatched.'} Screenshot refresh failed; inspect again before another action. "
                f"{type(exc).__name__}: {str(exc)[:180]}",
                status="uncertain" if action_succeeded or action_attempted else "failed",
                dispatched=True if action_succeeded else (None if action_attempted else False),
            )
