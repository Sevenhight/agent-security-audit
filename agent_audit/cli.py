"""CLI entry point.

    python -m agent_audit run --scenario indirect_injection
    python -m agent_audit run --scenario benign --mock
    python -m agent_audit list

``--mock`` replays a recorded message sequence so the pipeline can be
demonstrated without an API key. The mock script is read from
``examples/mock_scripts/<scenario>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .llm import DeepSeekClient, MockClient
from .llm_judge import triage
from .react_agent import Agent
from .report import AuditReport
from .rules import screen
from .scenarios import SCENARIOS, get
from .tools.registry import build_default_registry
from .trace import TraceRecorder, load_trace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "examples", "mock_scripts")


def _build_client(mock: bool, scenario_name: str):
    if not mock:
        return DeepSeekClient()
    script_path = os.path.join(MOCK_SCRIPT_DIR, f"{scenario_name}.json")
    if not os.path.isfile(script_path):
        raise SystemExit(f"[mock] missing script: {script_path}")
    with open(script_path, encoding="utf-8") as handle:
        return MockClient(json.load(handle))


def cmd_run(args: argparse.Namespace) -> int:
    scenario = get(args.scenario)
    run_id = args.run_id or f"{scenario.name}-{time.strftime('%Y%m%d-%H%M%S')}"

    workspace = scenario.materialize(PROJECT_ROOT)
    registry = build_default_registry(workspace)
    recorder = TraceRecorder(run_id=run_id, root=PROJECT_ROOT)
    client = _build_client(args.mock, scenario.name)

    print(f"[audit] scenario = {scenario.name} ({scenario.description})")
    print(f"[audit] workspace = {workspace}")
    print(f"[audit] trace     = {recorder.path}")

    agent = Agent(registry=registry, client=client, recorder=recorder, max_steps=args.max_steps)
    try:
        final_answer = agent.run(scenario.task)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and exit cleanly
        print(f"[audit] agent run failed: {exc}", file=sys.stderr)
        return 2

    events = load_trace(recorder.path)
    findings = screen(events)

    print(f"[audit] findings after rules: {len(findings)}")
    if args.no_triage:
        verdicts = []
    else:
        judge = _build_judge(mock=args.mock, scenario_name=scenario.name, client=client)
        verdicts = triage(findings, judge)

    report = AuditReport(
        run_id=run_id,
        task=scenario.task,
        trace_path=os.path.relpath(recorder.path, PROJECT_ROOT),
        trace_summary=recorder.summary(),
        findings=findings,
        verdicts=verdicts,
        final_answer=final_answer,
    )

    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, f"{run_id}.md")
    json_path = os.path.join(report_dir, f"{run_id}.json")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_markdown())
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_json())

    print()
    print(report.to_console())
    print()
    print(f"[audit] report -> {os.path.relpath(md_path, PROJECT_ROOT)}")
    print(f"[audit] json   -> {os.path.relpath(json_path, PROJECT_ROOT)}")
    if args.mock:
        print("[audit] mock mode: 结论由脚本化消息驱动，仅用于演示流程")

    return 0


def _build_judge(mock: bool, scenario_name: str, client):
    if not mock:
        return client
    script_path = os.path.join(MOCK_SCRIPT_DIR, f"{scenario_name}.judge.json")
    if not os.path.isfile(script_path):
        return MockClient([])
    with open(script_path, encoding="utf-8") as handle:
        return MockClient(json.load(handle))


def cmd_list(_: argparse.Namespace) -> int:
    print("available scenarios:")
    for name, scenario in SCENARIOS.items():
        expected = ", ".join(scenario.expect_categories) or "none (control group)"
        print(f"  {name:<22} {scenario.description}")
        print(f"  {'':<22} expected: {expected}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_audit",
        description="ReAct Agent 安全审计：轨迹采集 + 规则初筛 + 大模型研判",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="运行一次审计")
    run_parser.add_argument("--scenario", required=True, help="场景名，见 list 命令")
    run_parser.add_argument("--run-id", default=None, help="自定义 run id")
    run_parser.add_argument("--max-steps", type=int, default=8, help="Agent 最大步数")
    run_parser.add_argument("--mock", action="store_true", help="离线脚本模式，无需 API Key")
    run_parser.add_argument("--no-triage", action="store_true", help="跳过 LLM 二轮研判")
    run_parser.set_defaults(func=cmd_run)

    list_parser = sub.add_parser("list", help="列出全部场景")
    list_parser.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
