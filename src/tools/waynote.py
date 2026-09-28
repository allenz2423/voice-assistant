"""Waynote desktop sticky note manager.

Provides capabilities for creating, appending to, listing, and toggling
Wayland sticky notes via Waynote (dev.mryll.waynote).
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
VALID_COLORS = {"yellow", "green", "pink", "purple", "blue", "orange", "gray"}


def generate_ulid() -> str:
    """Generates a canonical 26-character Crockford Base32 ULID."""
    timestamp_ms = int(time.time() * 1000)
    time_chars = []
    for _ in range(10):
        time_chars.append(CROCKFORD_ALPHABET[timestamp_ms % 32])
        timestamp_ms //= 32
    time_part = "".join(reversed(time_chars))

    rand_bytes = os.urandom(16)
    rand_part = "".join(CROCKFORD_ALPHABET[b % 32] for b in rand_bytes)
    return time_part + rand_part


def get_notes_dir() -> Path:
    """Returns the notes directory path, ensuring it exists."""
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        base = Path(data_home)
    else:
        base = Path.home() / ".local" / "share"
    notes_dir = base / "waynote" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    return notes_dir


def _ensure_waynote_running() -> None:
    """Ensures the Waynote daemon is running to render sticky notes."""
    executable = shutil.which("waynote")
    if not executable:
        return

    # Check if waynote is already running
    try:
        res = subprocess.run(["pgrep", "-x", "waynote"], capture_output=True, check=False)
        if res.returncode == 0:
            # Tell existing instance to sync / show notes
            subprocess.run([executable, "show-all"], capture_output=True, check=False, timeout=3)
            return
    except Exception:
        pass

    # Launch detached waynote instance
    try:
        subprocess.Popen(
            [executable],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=os.environ,
        )
    except Exception:
        pass


def create_waynote(content: str, title: str = "", color: str = "yellow") -> str:
    """Creates a new Waynote sticky note on the desktop.

    Args:
        content: The text, markdown, or task items to put in the note.
        title: Optional title for the note header.
        color: Sticky note color ('yellow', 'green', 'pink', 'purple', 'blue', 'orange').
    """
    notes_dir = get_notes_dir()
    notes_dir.mkdir(parents=True, exist_ok=True)
    ulid = generate_ulid()

    chosen_color = color.lower().strip() if color else "yellow"
    if chosen_color not in VALID_COLORS:
        chosen_color = "yellow"

    clean_title = title.strip()
    if clean_title:
        # Create safe slug for filename
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean_title).strip("-").lower()
        if not slug:
            slug = "note"
    else:
        slug = "note"

    filename = f"{ulid}-{slug}.md"
    filepath = notes_dir / filename

    clean_content = content.strip()
    body_parts = []
    if clean_title:
        body_parts.append(f"# {clean_title}\n")
    if clean_content:
        body_parts.append(clean_content)

    body = "\n".join(body_parts) if body_parts else ""

    markdown_file = f"""---
id: {ulid}
color: {chosen_color}
pinned: false
locked: false
layer: front
tags: []
---

{body}
"""

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_file)
    except Exception as e:
        return f"Failed to save sticky note: {e}"

    _ensure_waynote_running()
    display_name = f"'{clean_title}'" if clean_title else "sticky note"
    return f"Created {chosen_color} {display_name} with {len(clean_content)} characters."


def append_waynote(content: str, target: str = "") -> str:
    """Appends text or checklist items to an existing sticky note.

    Args:
        content: Text or checklist items to append.
        target: Optional note title or ID substring. If omitted, appends to the most recent note.
    """
    notes_dir = get_notes_dir()
    note_files = [f for f in notes_dir.glob("*.md") if not f.name.endswith(".conflict.md")]
    if not note_files:
        return create_waynote(content=content, title=target)

    matched_file: Optional[Path] = None

    if target:
        target_lower = target.lower()
        for f in note_files:
            if target_lower in f.stem.lower():
                matched_file = f
                break

    if not matched_file:
        # Pick the most recently modified note
        note_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        matched_file = note_files[0]

    try:
        with open(matched_file, "r", encoding="utf-8") as f:
            existing = f.read()

        clean_add = content.strip()
        if not clean_add:
            return "No content provided to append."

        # Format as list item if user asked to add a task/item
        if clean_add.startswith(("-", "*", "•", "1.")):
            formatted = f"\n{clean_add}"
        else:
            formatted = f"\n- {clean_add}"

        with open(matched_file, "a", encoding="utf-8") as f:
            f.write(formatted)

        _ensure_waynote_running()
        return f"Appended to sticky note ({matched_file.stem.split('-', 1)[-1]}): {clean_add}"
    except Exception as e:
        return f"Failed to append to sticky note: {e}"


def list_waynotes() -> str:
    """Lists all current Waynote sticky notes and their contents."""
    notes_dir = get_notes_dir()
    note_files = [f for f in notes_dir.glob("*.md") if "conflict" not in f.name]
    if not note_files:
        return "No sticky notes found."

    note_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    summaries = []

    for f in note_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                text = fp.read()
            # Extract title and snippet
            lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("---")]
            # Filter frontmatter lines
            body_lines = [l for l in lines if not any(l.startswith(k) for k in ["id:", "color:", "pinned:", "locked:", "layer:", "tags:"])]
            preview = " ".join(body_lines[:2])[:120] if body_lines else "(empty)"
            summaries.append(f"- {f.stem.split('-', 1)[-1]}: {preview}")
        except Exception:
            continue

    if not summaries:
        return "No readable sticky notes found."

    return "Sticky notes:\n" + "\n".join(summaries)


def manage_waynote(action: str = "show-all") -> str:
    """Controls the Waynote application state.

    Args:
        action: One of 'show-all', 'hide-all', 'toggle', or 'new'.
    """
    executable = shutil.which("waynote")
    if not executable:
        return "Waynote is not installed on this system."

    valid_actions = {"show-all", "hide-all", "toggle", "new"}
    clean_action = action.lower().strip()
    if clean_action not in valid_actions:
        return f"Invalid action '{action}'. Choose from: show-all, hide-all, toggle, new."

    try:
        proc = subprocess.run(
            [executable, clean_action],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            return f"Waynote {clean_action} failed: {err}"
        return f"Waynote {clean_action} executed."
    except Exception as e:
        return f"Failed to execute waynote {clean_action}: {e}"
