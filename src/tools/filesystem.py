"""Small, explicit text-file operations for desktop tasks and readback verification."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def _target(path: str) -> Path:
    value = Path(os.path.expandvars(os.path.expanduser(str(path or ""))))
    if not str(path or "").strip():
        raise ValueError("A file path is required.")
    return value.resolve(strict=False)


def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a bounded UTF-8 text file for inspection or independent verification."""
    target = _target(path)
    info = target.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"'{target}' is not a regular file.")
    limit = min(max(int(max_chars), 1), 64000)
    with target.open("rb") as stream:
        data = stream.read(limit + 1)
    content = data[:limit].decode("utf-8", errors="replace")
    truncated = len(data) > limit or info.st_size > limit
    suffix = "\n[truncated]" if truncated else ""
    return f"File: {target}\nBytes: {info.st_size}\nText:\n{content}{suffix}"


def create_file(path: str, content: str) -> str:
    """Create a new UTF-8 text file without ever replacing an existing path."""
    target = _target(path)
    payload = str(content).encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return f"Created {target} ({len(payload)} bytes)."


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a file, or atomically replace it only when overwrite=True."""
    target = _target(path)
    if not target.exists():
        return create_file(str(target), content)
    if not overwrite:
        raise FileExistsError(f"'{target}' already exists; refusing to overwrite it.")
    if not target.is_file():
        raise ValueError(f"'{target}' is not a regular file.")
    mode = stat.S_IMODE(target.stat().st_mode)
    payload = str(content).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return f"Updated {target} ({len(payload)} bytes)."
