"""Structured trace recording.

Every event is one JSON line, which makes the file appendable during a live
run, greppable afterwards, and portable between machines. Four event families
are recorded: ``user_input``, ``model_reasoning``, ``tool_call`` and
``final_answer`` (plus error events).

The recorder also keeps an in-memory copy so the auditors can work on the run
without re-reading the file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

TRACE_DIR = "traces"


@dataclass
class TraceEvent:
    event_type: str
    timestamp: float
    step: int | None = None
    content: str | None = None
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None
    ok: bool | None = None
    error: str | None = None
    memory: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TraceRecorder:
    """Append-only JSONL writer with an in-memory mirror."""

    def __init__(self, run_id: str, root: str = ".") -> None:
        self.run_id = run_id
        directory = os.path.join(root, TRACE_DIR)
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{run_id}.jsonl")
        self.events: list[TraceEvent] = []
        # Truncate so a re-run with the same id never mixes two sessions.
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("")

    def record(self, event_type: str, **payload: Any) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            timestamp=time.time(),
            **{key: value for key, value in payload.items() if key in TraceEvent.__annotations__},
        )
        self.events.append(event)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")
        return event

    def write_memory(self, key: str, value: Any) -> None:
        """Record a memory write so the audit can see what the agent retained."""
        self.record("memory_write", memory={key: value})

    def summary(self) -> dict[str, Any]:
        tool_calls = [event for event in self.events if event.event_type == "tool_call"]
        return {
            "run_id": self.run_id,
            "path": self.path,
            "total_events": len(self.events),
            "tool_calls": len(tool_calls),
            "failed_calls": sum(1 for event in tool_calls if not event.ok),
            "tools_used": sorted({event.tool for event in tool_calls if event.tool}),
        }


def load_trace(path: str) -> list[dict[str, Any]]:
    """Read a JSONL trace back into plain dicts."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
