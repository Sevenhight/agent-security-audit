"""CLI entry point.

    python -m agent_audit run --scenario indirect_injection
    python -m agent_audit run --scenario benign --mock
    python -m agent_audit list

``--mock`` replays a recorded message sequence so the pipeline can be
demonstrated without an API key. The mock script is read from
``examples/mock_scripts/<scenario>.json``.

================================================================================
【这个文件是"主流程串联图"，面试前读一遍就能想起整条链路】
================================================================================

    cmd_run() 里按顺序做了七件事：

        ① get(scenario)             取场景配置
        ② scenario.materialize()    把沙箱文件写到磁盘
        ③ build_default_registry()  建工具集（绑定 workspace）
        ④ TraceRecorder(...)        建轨迹记录器
        ⑤ _build_client(mock)       建模型客户端（真实 API 或回放脚本）
        ⑥ Agent(...).run(task)      ★ 跑 Agent，产出轨迹
        ⑦ load_trace → screen       ★ 读回轨迹，跑规则
           → triage                 模型复核
           → AuditReport            出报告（md + json + 控制台）

    注意 ⑥ 和 ⑦ 之间的边界：
        Agent 跑完之后，审计**重新从文件里读轨迹**，而不是直接用 recorder.events。
        为什么？
            第一，验证"文件里存的信息足够重建审计"；
            第二，让在线审计和离线回放走同一条代码路径（都是 dict 列表）。
        这个设计保证了"事后重放结论一致"。

【★ 一个可以主动讲的架构优点】
    Agent 和审计器之间没有直接引用，只通过轨迹文件耦合。
    这意味着：
        · 审计器可以用来审任何能产出同格式轨迹的 Agent（不只是本项目这个）
        · Agent 可以换成 LangChain / LangGraph 实现，审计器不用改
    这个解耦是"审计工具"和"demo 脚本"的分水岭。
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

# 项目根目录 = cli.py 的上两级。
# 用它来定位 examples/ 和决定 traces/ reports/ 落在哪。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "examples", "mock_scripts")


def _build_client(mock: bool, scenario_name: str):
    """建模型客户端。

    【--mock 到底是什么】
        它把模型的回复**录成一个 JSON 数组**，按顺序回放：
            examples/mock_scripts/<scenario>.json        → Agent 的回复序列
            examples/mock_scripts/<scenario>.judge.json  → 复核员的回复序列

        用途：演示流水线 + CI 回归（不需要 API Key，结果确定）。
        ★ 局限：它验证的是"流程跑得通"，不是"模型判断得准"。
          CLI 输出末尾会打印提示，README 的已知局限里也写了。

        【为什么这个区分很重要】
            很多项目把 mock 结果当成效果证明，这是自欺。
            测试替身（test double）能验证流程，不能验证效果。
            面试时主动说清这一点，比含糊带过更显专业。

    【MockClient 是怎么实现的】
        见 llm.py：它内部有个游标，每 chat() 一次就吐出数组里的下一条。
        数组用完了会怎样？需要看 llm.py 的实现 —— 这是"读代码要读到底"的例子。
    """
    if not mock:
        # 真实模式：从环境变量读 DEEPSEEK_API_KEY
        return DeepSeekClient()
    script_path = os.path.join(MOCK_SCRIPT_DIR, f"{scenario_name}.json")
    if not os.path.isfile(script_path):
        raise SystemExit(f"[mock] missing script: {script_path}")
    with open(script_path, encoding="utf-8") as handle:
        return MockClient(json.load(handle))


def cmd_run(args: argparse.Namespace) -> int:
    """执行一次完整审计。返回值是退出码（0 成功，2 失败）。"""
    scenario = get(args.scenario)
    # run_id 默认带时间戳，保证多次运行不覆盖。
    # 用 --run-id 可以指定固定名字（比如 CI 里想要稳定的产物路径）。
    run_id = args.run_id or f"{scenario.name}-{time.strftime('%Y%m%d-%H%M%S')}"

    # ② 落盘沙箱。返回 workspace 路径，后面的工具都绑在这个目录上。
    workspace = scenario.materialize(PROJECT_ROOT)
    registry = build_default_registry(workspace)
    recorder = TraceRecorder(run_id=run_id, root=PROJECT_ROOT)
    client = _build_client(args.mock, scenario.name)

    # 这三行打印是给用户看的"我在干什么"，也是排障的第一手信息
    print(f"[audit] scenario = {scenario.name} ({scenario.description})")
    print(f"[audit] workspace = {workspace}")
    print(f"[audit] trace     = {recorder.path}")

    agent = Agent(registry=registry, client=client, recorder=recorder, max_steps=args.max_steps)
    try:
        final_answer = agent.run(scenario.task)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and exit cleanly
        # 【为什么这里宽泛捕获所有异常】
        #   CLI 是用户交互的边界。模型挂了（key 错/欠费/网络断）时，
        #   用户需要看到一句人话，而不是 20 行 traceback。
        #   注意：recorder 已经把 llm_error 事件写进轨迹了，
        #   所以即使这里退出，痕迹也留下来了。
        print(f"[audit] agent run failed: {exc}", file=sys.stderr)
        return 2

    # ---- 审计阶段：从文件读回轨迹，不直接用 recorder.events ----
    # 这个选择保证了在线审计和离线回放是同一份逻辑（见文件头说明）
    events = load_trace(recorder.path)
    findings = screen(events)

    print(f"[audit] findings after rules: {len(findings)}")
    if args.no_triage:
        # --no-triage 用于调试规则本身：跳过模型，只看规则召回了什么。
        # 这时候所有 finding 都是"未复核"状态，会被全部保留。
        verdicts = []
    else:
        judge = _build_judge(mock=args.mock, scenario_name=scenario.name, client=client)
        verdicts = triage(findings, judge)

    report = AuditReport(
        run_id=run_id,
        task=scenario.task,
        # 存相对路径，报告跟着仓库走，换机器也能对上
        trace_path=os.path.relpath(recorder.path, PROJECT_ROOT),
        trace_summary=recorder.summary(),
        findings=findings,
        verdicts=verdicts,
        final_answer=final_answer,
    )

    # ---- 双产物：md 给人看，json 给机器读 ----
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
        # ★ 这句提示不能省。不打印的话，看输出的人会以为这是真实效果。
        print("[audit] mock mode: 结论由脚本化消息驱动，仅用于演示流程")

    return 0


def _build_judge(mock: bool, scenario_name: str, client):
    """建复核员客户端。

    【mock 模式下为什么是两个独立的脚本文件】
        Agent 的回复和复核员的回复是两套不同的调用序列，
        混在一个数组里会互相干扰。

        另一个原因：benign 场景的规则命中数是 0，
        根本不会调用复核员，所以它没有 .judge.json ——
        这里用 MockClient([]) 兜底（空脚本，永远不吐东西）。
    """
    if not mock:
        return client
    script_path = os.path.join(MOCK_SCRIPT_DIR, f"{scenario_name}.judge.json")
    if not os.path.isfile(script_path):
        return MockClient([])
    with open(script_path, encoding="utf-8") as handle:
        return MockClient(json.load(handle))


def cmd_list(_: argparse.Namespace) -> int:
    """列场景。第一次跑必用，也是 `get()` 报错时的补充信息。"""
    print("available scenarios:")
    for name, scenario in SCENARIOS.items():
        expected = ", ".join(scenario.expect_categories) or "none (control group)"
        print(f"  {name:<22} {scenario.description}")
        print(f"  {'':<22} expected: {expected}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义。

    【为什么用 subcommand 而不是一堆 flag】
        argparse 的 subparsers 让 "run" 和 "list" 各有各的参数空间，
        不会出现 "list --scenario xxx" 这种没意义的组合。
        这是 CLI 设计的基本规范（git / docker / kubectl 都是这样）。

    【每个参数存在的理由】
        --scenario   必填，没有它不知道跑什么
        --run-id     可选，CI 里需要稳定的产物文件名时用
        --max-steps  默认 8，改小可以测试预算耗尽（R-07）
        --mock       不用 API Key 演示全流程
        --no-triage  跳过模型复核，只看规则召回（调规则时用）
    """
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
    # set_defaults(func=...) 把"这个子命令该调哪个函数"挂在解析结果上，
    # main() 里就不用写 if/elif 分支了 —— 加子命令时只改这里。
    run_parser.set_defaults(func=cmd_run)

    list_parser = sub.add_parser("list", help="列出全部场景")
    list_parser.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """入口。

    argv 参数默认 None → argparse 会去读 sys.argv。
    允许传参是为了测试时可以模拟命令行调用，不用改 sys.argv。
    （这个技巧在所有 CLI 项目里都值得用）
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
