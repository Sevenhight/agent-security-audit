"""HTTP tool used as the agent's egress channel.

Two security-relevant behaviours are deliberately included because they are
the exact primitives the audit is meant to catch:

1. The response body is fed back into the model verbatim (truncated), which
   makes it an indirect prompt-injection vector.
2. Localhost / metadata endpoints are *not* blocked by default, mimicking a
   naively configured production agent. ``ALLOW_PRIVATE_TARGETS`` lets the
   demo toggle a mitigation on.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

MAX_BODY_BYTES = 8 * 1024
TIMEOUT_SECONDS = 10

# Flip to "0" to demonstrate the egress allow-list mitigation in the report.
ALLOW_PRIVATE_TARGETS = os.environ.get("ALLOW_PRIVATE_TARGETS", "1") == "1"


def http_get(url: str) -> str:
    if not ALLOW_PRIVATE_TARGETS and _looks_private(url):
        return f"[http_get] blocked by egress policy: {url}"

    request = urllib.request.Request(url, headers={"User-Agent": "agent-audit-demo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
            return f"[http_get] {response.status} {url}\n{body}"
    except urllib.error.HTTPError as exc:
        return f"[http_get] HTTP {exc.code} {url}"
    except Exception as exc:  # noqa: BLE001 - demo tool, surface the reason
        return f"[http_get] error {url}: {exc}"


def _looks_private(url: str) -> bool:
    lowered = url.lower()
    private_markers = (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",  # cloud metadata
        "10.",
        "192.168.",
    )
    return any(marker in lowered for marker in private_markers)
