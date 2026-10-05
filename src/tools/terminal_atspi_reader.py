"""System-Python helper for reading an accessible terminal text widget."""

from __future__ import annotations

import json
import sys
from typing import Any


def _focused(state_set: Any, atspi: Any) -> bool:
    try:
        return bool(state_set.contains(atspi.StateType.FOCUSED))
    except Exception:
        return False


def _read_text_range(atspi: Any, node: Any, start: int, end: int) -> str:
    """Read text through the AT-SPI Text interface, not Accessible.get_text()."""
    return atspi.Text.get_text(node, start, end) or ""


def read_terminal(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        Atspi.init()
        desktop = Atspi.get_desktop(0)
        if payload.get("find_focused_terminal"):
            result: dict[str, Any] = {}
            visited = 0

            def find_focused(node: Any, depth: int = 0) -> None:
                nonlocal visited, result
                if result or visited >= 1200 or depth > 18:
                    return
                visited += 1
                try:
                    role = (node.get_role_name() or "").casefold()
                    state = node.get_state_set()
                    has_text = node.get_text_iface()
                    if ("terminal" in role and has_text
                            and Atspi.Text.get_character_count(node) and _focused(state, Atspi)):
                        app_node = node
                        title = node.get_name() or ""
                        for _ in range(18):
                            parent = app_node.get_parent()
                            if not parent:
                                break
                            app_node = parent
                            parent_role = (app_node.get_role_name() or "").casefold()
                            if parent_role in {"frame", "window", "application"} and app_node.get_name():
                                title = app_node.get_name()
                            if parent_role == "application":
                                break
                        try:
                            app_name = app_node.get_name() or ""
                            pid = int(app_node.get_process_id() or node.get_process_id() or 0)
                        except Exception:
                            app_name, pid = "", 0
                        if pid:
                            result = {"ok": True, "pid": pid, "app": app_name, "title": title}
                            return
                    for index in range(min(node.get_child_count(), 80)):
                        child = node.get_child_at_index(index)
                        if child:
                            find_focused(child, depth + 1)
                        if result:
                            return
                except Exception:
                    return

            find_focused(desktop)
            return result or {"ok": False, "error": "No focused accessible terminal widget was found."}

        pid = int(payload.get("pid") or 0)
        title = str(payload.get("title") or "").casefold()
        app = next(
            (
                desktop.get_child_at_index(i)
                for i in range(desktop.get_child_count())
                if desktop.get_child_at_index(i)
                and desktop.get_child_at_index(i).get_process_id() == pid
            ),
            None,
        )
        if app is None:
            return {"ok": False, "error": "The focused terminal is not registered on the AT-SPI bus."}

        roots = [app.get_child_at_index(i) for i in range(app.get_child_count())]
        roots = [node for node in roots if node]
        if title:
            matching = [node for node in roots if title in (node.get_name() or "").casefold()]
            roots = matching or roots

        candidates: list[tuple[int, int, Any, Any, str]] = []
        visited = 0

        def visit(node: Any, depth: int = 0) -> None:
            nonlocal visited
            if visited >= 300 or depth > 12:
                return
            visited += 1
            try:
                role = (node.get_role_name() or "").casefold()
                has_text = node.get_text_iface()
                if has_text and "terminal" in role:
                    count = int(Atspi.Text.get_character_count(node) or 0)
                    if count:
                        state = node.get_state_set()
                        name = node.get_name() or ""
                        candidates.append((1 if _focused(state, Atspi) else 0, count, node, role, name))
                for index in range(min(node.get_child_count(), 80)):
                    child = node.get_child_at_index(index)
                    if child:
                        visit(child, depth + 1)
            except Exception:
                return

        for root in roots:
            visit(root)
        if not candidates:
            return {"ok": False, "error": "No accessible terminal text widget was found."}

        # If a terminal embeds multiple surfaces, prefer the one with keyboard
        # focus; otherwise prefer the largest text range.
        _is_focused, count, text_node, role, name = max(candidates, key=lambda row: (row[0], row[1]))
        if payload.get("probe_only"):
            return {"ok": True, "role": role, "name": name, "accessible_characters": count}
        scope = str(payload.get("scope") or "screen")
        max_chars = max(1, min(int(payload.get("max_chars") or 12000), 24000))
        line_limit = {"screen": 80, "recent": 200}.get(scope)
        # Fetch from the tail only, so a long scrollback cannot create an
        # unbounded allocation in the daemon or helper process.
        start = max(0, count - max_chars - 4096)
        text = _read_text_range(Atspi, text_node, start, count)
        truncated = start > 0
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if line_limit is not None:
            lines = text.splitlines()
            if len(lines) > line_limit:
                text = "\n".join(lines[-line_limit:])
                truncated = True
        if len(text) > max_chars:
            text = text[-max_chars:]
            truncated = True
        return {"ok": True, "text": text, "role": role, "name": name,
                "truncated": truncated, "accessible_characters": count}
    except Exception as exc:
        return {"ok": False, "error": f"AT-SPI unavailable: {type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    try:
        request = json.loads(sys.argv[1])
        response = read_terminal(request)
    except Exception as exc:
        response = {"ok": False, "error": str(exc)}
    print(json.dumps(response, ensure_ascii=False))
