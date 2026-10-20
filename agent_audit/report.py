"""Audit report generation.

The report is the deliverable an analyst actually reads, so it is ordered by
what the reader needs first: a verdict, then the attack chain, then per-finding
detail with a concrete fix. Every finding carries the trace step it came from,
so a reader can jump straight from the conclusion to the evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm_judge import Verdict
from .rules import SEVERITY_ORDER, Finding

REMEDIATION = {
    "prompt_injection": [
        "对工具返回值做来源标记，在上下文中显式标注为「不可信数据」，禁止其中的指令被当作任务执行",
        "在系统提示中加入注入检测前置指令，并对高危动作增加二次确认环节",
    ],
    "privilege_escalation": [
        "为写操作、网络出口类工具配置白名单与人工确认，禁止模型自主触发",
        "按最小权限拆分工具集：读任务的会话不下发写工具",
    ],
    "sensitive_data_access": [
        "对文件工具做路径规范化与根目录校验，SQL 工具强制只读连接",
        "对返回值做敏感信息脱敏（密钥、凭据、身份证号）后再进入上下文",
    ],
    "data_exfiltration": [
        "配置出口域名白名单，阻断外部回传地址",
        "对出站内容做敏感字段扫描，命中即阻断并告警",
    ],
    "hallucination": [
        "工具调用前校验工具名与参数 schema，失败时返回结构化错误而非静默降级",
        "为 Agent 设置步数与耗时上限，超限即中止并回报",
    ],
}

CATEGORY_LABEL = {
    "prompt_injection": "提示词注入",
    "privilege_escalation": "越权工具调用",
    "sensitive_data_access": "敏感数据访问",
    "data_exfiltration": "数据外发",
    "hallucination": "幻觉与异常行为",
}


@dataclass
class AuditReport:
    run_id: str
    task: str
    trace_path: str
    trace_summary: dict[str, Any]
    findings: list[Finding]
    verdicts: list[Verdict]
    final_answer: str = ""
    notes: list[str] = field(default_factory=list)

    # --- Derived views ---------------------------------------------------

    @property
    def kept(self) -> list[tuple[Finding, Verdict]]:
        index = {verdict.rule_id: verdict for verdict in self.verdicts}
        pairs = []
        for finding in self.findings:
            verdict = index.get(finding.rule_id)
            if verdict is None or verdict.kept:
                pairs.append(
                    (
                        finding,
                        verdict
                        or Verdict(finding.rule_id, "uncertain", 0.0, "未复核", "未知"),
                    )
                )
        return pairs

    @property
    def risk_level(self) -> str:
        if not self.kept:
            return "低"
        top = max(SEVERITY_ORDER[finding.severity] for finding, _ in self.kept)
        return {4: "严重", 3: "高", 2: "中", 1: "低", 0: "信息"}[top]

    def chain(self) -> list[str]:
        """Human-readable attack chain, ordered by trace step."""
        steps = []
        for finding, verdict in sorted(self.kept, key=lambda pair: pair[0].step or 0):
            label = CATEGORY_LABEL.get(finding.category, finding.category)
            steps.append(f"step {finding.step} · {label} · {finding.detail}")
            if verdict.verdict == "true_positive":
                steps.append(f"        └ 确认：{verdict.reason}")
        return steps

    # --- Renderers -------------------------------------------------------

    def to_markdown(self) -> str:
        lines = [
            f"# Agent 安全审计报告 · {self.run_id}",
            "",
            f"- **审计任务**：{self.task}",
            f"- **轨迹文件**：`{self.trace_path}`",
            f"- **工具调用**：{self.trace_summary.get('tool_calls', 0)} 次"
            f"（失败 {self.trace_summary.get('failed_calls', 0)} 次）",
            f"- **规则告警**：{len(self.findings)} 条，复核后保留 {len(self.kept)} 条",
            f"- **综合风险等级**：**{self.risk_level}**",
            "",
            "## 一、结论",
            "",
        ]
        if self.kept:
            confirmed = sum(1 for _, verdict in self.kept if verdict.verdict == "true_positive")
            lines.append(
                f"本次运行检出 **{len(self.kept)}** 条风险，其中 **{confirmed}** 条经复核确认为真实风险。"
                f"整体风险等级评定为 **{self.risk_level}**。"
            )
        else:
            lines.append("本次运行未检出风险，规则告警经复核均为误报或证据不足。")

        lines.extend(["", "## 二、攻击链还原", ""])
        if self.kept:
            for step in self.chain():
                lines.append(step if step.startswith("    ") else f"- {step}")
        else:
            lines.append("无可还原的攻击链。")

        lines.extend(["", "## 三、风险明细", ""])
        for index, (finding, verdict) in enumerate(self.kept, start=1):
            label = CATEGORY_LABEL.get(finding.category, finding.category)
            lines.extend(
                [
                    f"### {index}. [{finding.severity.upper()}] {label} · {finding.rule_id}",
                    "",
                    f"- **触发位置**：trace step {finding.step}",
                    f"- **检测说明**：{finding.detail}",
                    f"- **命中内容**：`{finding.matched}`",
                    f"- **复核结论**：{verdict.verdict}（置信度 {verdict.confidence:.2f}）",
                    f"- **复核理由**：{verdict.reason}",
                    f"- **实际影响**：{verdict.impact}",
                    "",
                    "<details><summary>证据</summary>",
                    "",
                    "```",
                    finding.evidence,
                    "```",
                    "",
                    "</details>",
                    "",
                    "**修复建议**",
                    "",
                ]
            )
            for item in REMEDIATION.get(finding.category, ["人工评估该风险并补充控制措施"]):
                lines.append(f"- {item}")
            lines.append("")

        if self.notes:
            lines.extend(["## 四、备注", ""])
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "*本报告由 agent-audit 自动生成：规则引擎完成初筛召回，大模型完成二轮研判，"
                "所有结论均可回溯到上文的 trace 步骤。*",
            ]
        )
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "run_id": self.run_id,
            "task": self.task,
            "trace_path": self.trace_path,
            "risk_level": self.risk_level,
            "final_answer": self.final_answer,
            "trace_summary": self.trace_summary,
            "findings": [
                {
                    **finding.to_dict(),
                    "verdict": {
                        "verdict": verdict.verdict,
                        "confidence": verdict.confidence,
                        "reason": verdict.reason,
                        "impact": verdict.impact,
                    },
                }
                for finding, verdict in self.kept
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_console(self) -> str:
        lines = [
            f"run_id        : {self.run_id}",
            f"risk level    : {self.risk_level}",
            f"tool calls    : {self.trace_summary.get('tool_calls', 0)}"
            f" (failed {self.trace_summary.get('failed_calls', 0)})",
            f"findings      : {len(self.findings)} raw -> {len(self.kept)} kept",
        ]
        for finding, verdict in self.kept:
            lines.append(
                f"  [{finding.severity:>8}] step {finding.step} {finding.rule_id} "
                f"{finding.category} -> {verdict.verdict} ({verdict.confidence:.2f})"
            )
        return "\n".join(lines)
