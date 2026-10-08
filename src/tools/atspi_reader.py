"""System-Python helper: serialize requested windows' AT-SPI trees."""

import json
import sys


class _ScopeUnavailable(Exception):
    """The requested browser document cannot be identified safely."""


def _parent_path(node, ancestor, max_depth=64):
    """Return node's path below ancestor, or fail if it is not contained there."""
    reverse_path = []
    visited = set()
    current = node
    while current != ancestor:
        marker = id(current)
        if marker in visited or len(reverse_path) >= max_depth:
            raise _ScopeUnavailable("active descendant chain is cyclic or too deep")
        visited.add(marker)
        reverse_path.append(current)
        current = current.get_parent()
        if current is None:
            raise _ScopeUnavailable("active descendant escapes the matched window")
    reverse_path.reverse()
    return reverse_path


def _is_web_document(node, Atspi):
    try:
        return node.get_role() == Atspi.Role.DOCUMENT_WEB
    except Exception as exc:
        raise _ScopeUnavailable("could not determine accessible role") from exc


def _focused_children(node, Atspi):
    try:
        child_count = int(node.get_child_count())
        if child_count > 500:
            raise _ScopeUnavailable("focused descendant is ambiguous")
        focused = []
        for index in range(child_count):
            child = node.get_child_at_index(index)
            if child is None:
                continue
            state = child.get_state_set()
            if state is None:
                raise _ScopeUnavailable("could not determine descendant focus")
            if state.contains(Atspi.StateType.FOCUSED):
                focused.append(child)
        return focused
    except _ScopeUnavailable:
        raise
    except Exception as exc:
        raise _ScopeUnavailable("could not determine descendant focus") from exc


def _active_web_document(window, Atspi):
    """Resolve one web document through active/focused descendants only."""
    current = window
    visited = {id(window)}
    for _ in range(64):
        if _is_web_document(current, Atspi):
            if current == window:
                raise _ScopeUnavailable("matched window is not a web document")
            return current

        try:
            active = current.get_active_descendant()
        except Exception as exc:
            raise _ScopeUnavailable("active descendant is unavailable") from exc

        focused = _focused_children(current, Atspi) if active is None else []
        if active is not None:
            path = _parent_path(active, current)
            # If focus is also exposed at this level, it must identify the same
            # active branch; otherwise the document choice is ambiguous.
            focused = _focused_children(current, Atspi)
            if len(focused) > 1 or (focused and focused[0] != path[0]):
                raise _ScopeUnavailable("active and focused descendants disagree")
        else:
            if len(focused) != 1:
                raise _ScopeUnavailable("no unique focused descendant is available")
            active = focused[0]
            path = _parent_path(active, current)

        if not path:
            raise _ScopeUnavailable("active descendant chain made no progress")
        if any(id(item) in visited for item in path):
            raise _ScopeUnavailable("active descendant chain is cyclic")
        visited.update(id(item) for item in path)
        documents = [item for item in path if _is_web_document(item, Atspi)]
        if len(documents) > 1:
            raise _ScopeUnavailable("active descendant chain has multiple web documents")
        if documents:
            return documents[0]
        current = path[-1]
    raise _ScopeUnavailable("active descendant chain is too deep")


def _browser_window(app, title):
    """Require exactly one top-level accessible window with the exact title."""
    if not title:
        raise _ScopeUnavailable("active window title is unavailable")
    try:
        matches = []
        for index in range(int(app.get_child_count())):
            candidate = app.get_child_at_index(index)
            if candidate is None:
                continue
            role = (candidate.get_role_name() or "").casefold()
            name = (candidate.get_name() or "").strip()
            if role in {"frame", "window"} and name.casefold() == title.strip().casefold():
                matches.append(candidate)
    except Exception as exc:
        raise _ScopeUnavailable("could not identify the matched top-level window") from exc
    if len(matches) != 1:
        raise _ScopeUnavailable("no unique exact-title top-level window was found")
    return matches[0]


def read_tree(pid: int, title: str, browser_document_only: bool = False) -> dict:
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        Atspi.init()
        desktop = Atspi.get_desktop(0)
        apps = []
        for i in range(desktop.get_child_count()):
            candidate = desktop.get_child_at_index(i)
            if candidate and candidate.get_process_id() == pid:
                apps.append(candidate)
        if len(apps) != 1:
            if browser_document_only:
                return {"ok": False, "error": "Scope unavailable: no unique AT-SPI application matched the exact PID."}
            app = apps[0] if apps else None
        else:
            app = apps[0]
        if app is None:
            if browser_document_only:
                return {"ok": False, "error": "Scope unavailable: the exact PID is not registered on the AT-SPI bus."}
            return {"ok": False, "error": "The active application is not registered on the AT-SPI bus."}

        if browser_document_only:
            try:
                window = _browser_window(app, title)
                root = _active_web_document(window, Atspi)
            except _ScopeUnavailable as exc:
                return {"ok": False, "error": f"Scope unavailable: {exc}."}
        else:
            windows = [app.get_child_at_index(i) for i in range(app.get_child_count())]
            windows = [window for window in windows if window]
            root = next((w for w in windows if title and title.casefold() in (w.get_name() or "").casefold()), None)
            root = root or next((w for w in windows if w.get_role_name() == "frame"), app)
        lines, budget = [], [0, 0]

        def visit(node, depth=0):
            if budget[0] >= 12 or budget[1] >= 900 or depth > 6:
                return
            try:
                role, name = node.get_role_name() or "unknown", (node.get_name() or "").replace("\n", " ").strip()
                action = node.get_action_iface()
                actions = [action.get_action_name(i) for i in range(min(action.get_n_actions(), 4))] if action else []
                text = ""
                text_iface = node.get_text_iface()
                if text_iface and text_iface.get_character_count():
                    text = (text_iface.get_text(0, min(text_iface.get_character_count(), 180)) or "").strip()
                details = [f'"{name[:140]}"'] if name else []
                if text and text != name:
                    details.append(f"text: {text[:180]!r}")
                if actions:
                    details.append("actions: " + ", ".join(actions))
                line = "  " * depth + role + (" — " + " | ".join(details) if details else "")
                lines.append(line[:420])
                budget[0] += 1
                budget[1] += len(line)
                # Large lists often contain filenames or other unrelated data.
                # Read a small sample so each window gets represented fairly.
                child_limit = 8 if role.casefold() in {"list", "table", "tree", "tree table"} else 24
                for i in range(min(node.get_child_count(), child_limit)):
                    if budget[0] >= 12 or budget[1] >= 900:
                        break
                    child = node.get_child_at_index(i)
                    if child:
                        visit(child, depth + 1)
            except Exception as exc:
                if browser_document_only:
                    raise _ScopeUnavailable(
                        f"document traversal failed ({type(exc).__name__})"
                    ) from exc

        visit(root)
        result = {"ok": True, "application": app.get_name(), "window": root.get_name(),
                  "tree": "\n".join(lines)}
        if browser_document_only:
            result["scope"] = "web_document"
        return result
    except _ScopeUnavailable as exc:
        return {"ok": False, "error": f"Scope unavailable: {exc}."}
    except Exception as exc:
        if browser_document_only:
            return {"ok": False, "error": f"Scope unavailable: AT-SPI unavailable: {type(exc).__name__}: {exc}"}
        return {"ok": False, "error": f"AT-SPI unavailable: {exc}"}


if __name__ == "__main__":
    try:
        requested = json.loads(sys.argv[1])
        if isinstance(requested, list):
            result = [read_tree(int(item.get("pid", 0)), str(item.get("title") or ""),
                                bool(item.get("browser_document_only"))) for item in requested]
        else:
            result = read_tree(int(requested), sys.argv[2] if len(sys.argv) > 2 else "")
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False))
