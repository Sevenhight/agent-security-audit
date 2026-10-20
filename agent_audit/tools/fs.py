"""Filesystem tools, confined to a workspace root.

The point of this module is not to be a full-featured file API. It exists so
the audit demo has a realistic *sensitive* tool whose path handling can be
attacked. ``_resolve`` is the security boundary: everything outside the root
is rejected before any I/O happens.
"""

from __future__ import annotations

import os

MAX_READ_BYTES = 8 * 1024


def _resolve(workspace: str, path: str) -> str:
    """Resolve ``path`` inside ``workspace``, refusing traversal.

    We compare the *real* paths so that symlinks and ``..`` tricks cannot
    escape the sandbox. This is the kind of check that is easy to write
    incorrectly, which is why it is isolated in one function.
    """
    root = os.path.realpath(workspace)
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise PermissionError(f"path escapes workspace: {path}")
    return target


def read_file(workspace: str, path: str) -> str:
    target = _resolve(workspace, path)
    if not os.path.isfile(target):
        return f"[read_file] no such file: {path}"
    with open(target, "rb") as handle:
        data = handle.read(MAX_READ_BYTES)
    text = data.decode("utf-8", errors="replace")
    if os.path.getsize(target) > MAX_READ_BYTES:
        text += f"\n...[truncated at {MAX_READ_BYTES} bytes]"
    return text


def write_file(workspace: str, path: str, content: str) -> str:
    target = _resolve(workspace, path)
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"[write_file] wrote {len(content.encode('utf-8'))} bytes to {path}"
