"""System-Python helper: serialize requested windows' AT-SPI trees."""

import json
import sys


def read_tree(pid: int, title: str) -> dict:
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        Atspi.init()
        desktop = Atspi.get_desktop(0)
        app = None
        for i in range(desktop.get_child_count()):
            candidate = desktop.get_child_at_index(i)
            if candidate and candidate.get_process_id() == pid:
                app = candidate
                break
        if app is None:
            return {"ok": False, "error": "The active application is not registered on the AT-SPI bus."}

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
            except Exception:
                pass

        visit(root)
        return {"ok": True, "application": app.get_name(), "window": root.get_name(), "tree": "\n".join(lines)}
    except Exception as exc:
        return {"ok": False, "error": f"AT-SPI unavailable: {exc}"}


if __name__ == "__main__":
    try:
        requested = json.loads(sys.argv[1])
        if isinstance(requested, list):
            result = [read_tree(int(item.get("pid", 0)), str(item.get("title") or "")) for item in requested]
        else:
            result = read_tree(int(requested), sys.argv[2] if len(sys.argv) > 2 else "")
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False))
