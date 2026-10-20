"""Tests for the ReAct loop: parsing, dispatch, budgets and error paths."""

from __future__ import annotations

import json

from agent_audit.llm import MockClient
from agent_audit.react_agent import Agent, _parse_decision
from agent_audit.tools.registry import Tool, ToolRegistry
from agent_audit.trace import TraceRecorder


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="echo back",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text: f"echo:{text}",
        )
    )
    return registry


def _agent(tmp_path, scripted, max_steps=5) -> tuple[Agent, TraceRecorder]:
    recorder = TraceRecorder(run_id="test", root=str(tmp_path))
    agent = Agent(
        registry=_registry(),
        client=MockClient(scripted),
        recorder=recorder,
        max_steps=max_steps,
    )
    return agent, recorder


def test_parse_plain_json() -> None:
    payload, error = _parse_decision('{"type": "final", "answer": "ok"}')
    assert error is None and payload["answer"] == "ok"


def test_parse_fenced_json() -> None:
    payload, error = _parse_decision('```json\n{"type": "final", "answer": "ok"}\n```')
    assert error is None and payload["type"] == "final"


def test_parse_json_with_leading_prose() -> None:
    payload, error = _parse_decision('好的，结果如下：{"type": "final", "answer": "ok"}')
    assert error is None and payload["answer"] == "ok"


def test_parse_returns_error_without_json() -> None:
    payload, error = _parse_decision("我不知道该怎么回答")
    assert payload == {} and error == "no_json_object_found"


def test_agent_executes_tool_then_finishes(tmp_path) -> None:
    scripted = [
        json.dumps({"type": "action", "thought": "t", "tool": "echo", "arguments": {"text": "hi"}}),
        json.dumps({"type": "final", "thought": "done", "answer": "结果是 echo:hi"}),
    ]
    agent, recorder = _agent(tmp_path, scripted)
    assert agent.run("say hi") == "结果是 echo:hi"
    assert recorder.summary()["tool_calls"] == 1


def test_agent_records_unknown_tool(tmp_path) -> None:
    scripted = [
        json.dumps({"type": "action", "tool": "nope", "arguments": {}}),
        json.dumps({"type": "final", "answer": "gave up"}),
    ]
    agent, recorder = _agent(tmp_path, scripted)
    agent.run("x")
    failures = [e for e in recorder.events if e.event_type == "tool_call" and not e.ok]
    assert len(failures) == 1 and failures[0].error == "unknown_tool"


def test_agent_survives_malformed_output(tmp_path) -> None:
    scripted = [
        "这里没有 JSON",
        json.dumps({"type": "final", "answer": "recovered"}),
    ]
    agent, recorder = _agent(tmp_path, scripted)
    assert agent.run("x") == "recovered"
    assert any(e.event_type == "parse_error" for e in recorder.events)


def test_agent_respects_step_budget(tmp_path) -> None:
    loop = json.dumps({"type": "action", "tool": "echo", "arguments": {"text": "again"}})
    agent, recorder = _agent(tmp_path, [loop] * 10, max_steps=3)
    answer = agent.run("loop forever")
    assert answer.startswith("[budget_exhausted]")
    assert recorder.summary()["tool_calls"] == 3


def test_trace_file_is_valid_jsonl(tmp_path) -> None:
    scripted = [json.dumps({"type": "final", "answer": "ok"})]
    agent, recorder = _agent(tmp_path, scripted)
    agent.run("x")
    with open(recorder.path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert lines and all("event_type" in line for line in lines)
