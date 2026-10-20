"""End-to-end regression over the built-in scenarios.

These run the real CLI in mock mode, so they cover the trace -> rules ->
triage -> report path end to end without an API key. The control scenario
asserts zero findings: a rule set that only ever grows is a rule set that
will eventually drown the analyst in noise.
"""

from __future__ import annotations

import os

from agent_audit.cli import main
from agent_audit.scenarios import SCENARIOS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_control_scenario_has_no_findings() -> None:
    exit_code = main(["run", "--scenario", "benign", "--mock", "--run-id", "test-benign"])
    assert exit_code == 0
    report = os.path.join(PROJECT_ROOT, "reports", "test-benign.md")
    with open(report, encoding="utf-8") as handle:
        content = handle.read()
    assert "未检出风险" in content


def test_injection_scenario_reaches_critical() -> None:
    exit_code = main(
        ["run", "--scenario", "indirect_injection", "--mock", "--run-id", "test-inject"]
    )
    assert exit_code == 0
    with open(os.path.join(PROJECT_ROOT, "reports", "test-inject.md"), encoding="utf-8") as handle:
        content = handle.read()
    assert "严重" in content
    assert "privilege_escalation" not in content or "越权工具调用" in content


def test_every_scenario_runs() -> None:
    for name in SCENARIOS:
        assert main(["run", "--scenario", name, "--mock", "--run-id", f"test-{name}"]) == 0
