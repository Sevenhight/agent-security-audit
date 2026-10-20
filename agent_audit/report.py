"""Audit report generation.

The report is the deliverable an analyst actually reads, so it is ordered by
what the reader needs first: a verdict, then the attack chain, then per-finding
detail with a concrete fix. Every finding carries the trace step it came from,
so a reader can jump straight from the conclusion to the evidence.

================================================================================
【报告的章节顺序 = 分析员的阅读顺序，不是作者的写作顺序】
================================================================================

    一、结论        先回答"有没有事、多严重"（3 秒能看完）
    二、攻击链还原   再回答"发生了什么"（时间线，30 秒能看完）
    三、风险明细     最后才回答"具体哪一步、证据是什么、怎么修"

    为什么这个顺序重要？
        安全报告最常见的失败是"先讲技术细节，结论埋在最后"。
        决策者只想看第一段，工程同学才看第三段。
        一份报告要同时服务两类读者，顺序就必须是倒金字塔。

【★ 修复建议里最重要的一条，不是"加固检测"，是"最小权限"】

    看 REMEDIATION 里的建议，注意它们的方向：
        不是"提高规则准确率"，而是"让这类操作根本做不到"。
        - 写操作、网络出口类工具配置白名单与人工确认
        - 按最小权限拆分工具集：读任务的会话不下发写工具

    这是安全工程里最重要的一条原则：
        **检测是兜底，权限才是防线。**
    检测一定有漏，权限做对了漏了也没事。

    面试如果被问"你的方案能百分百防住注入吗"，正确回答是：
        不能，而且没有任何检测方案能。所以报告里给的建议
        主要落在权限和架构上，检测只是最后一层。
        这个回答能体现你理解"纵深防御"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm_judge import Verdict
from .rules import SEVERITY_ORDER, Finding

# 分类 → 修复建议。
#
# 【为什么用 dict 而不是在 Finding 里带建议】
#   同一类风险的修复建议是共享的，放在一处便于统一维护。
#   如果每条 finding 各带一份，改一次建议要改所有地方。
#
# 【建议的排列顺序有讲究：先架构、后检测】
#   每条列表的第一个都是"让这件事做不到"，第二个才是"检测它"。
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

# 英文 category → 中文标签。
# 为什么不直接把中文写进 Finding.category？
#   因为 category 是程序用的标识（测试断言、统计、筛选都靠它），
#   必须是稳定的 ASCII。中文只用于展示。
#   "内部标识与展示文案分离"是国际化支持的最小雏形。
CATEGORY_LABEL = {
    "prompt_injection": "提示词注入",
    "privilege_escalation": "越权工具调用",
    "sensitive_data_access": "敏感数据访问",
    "data_exfiltration": "数据外发",
    "hallucination": "幻觉与异常行为",
}


@dataclass
class AuditReport:
    """一份审计报告。

    【findings 和 verdicts 分开存，而不是合并成一个列表】
        因为 triage() 返回的 verdicts 不带 finding 引用，只带 rule_id。
        这个设计有个已知缺陷：如果两个 finding 的 rule_id 相同
        （比如两条 R-03），index 字典会让后面的覆盖前面的。

        实际影响：只影响 verdict 到 finding 的映射，
        不影响 findings 本身的数量和内容。
        更好的设计是给每条 finding 一个唯一 id，verdict 引用它。
        （这个点可以在"下一步改进"里主动提，显得你想得深）
    """

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
        """复核后保留的告警（finding + verdict 配对）。

        【注意 None 的处理】
            如果某条 finding 找不到对应 verdict（比如 --no-triage 模式），
            就给它一个"未复核"的占位 Verdict，
            而 Verdict.kept 对 "uncertain" 返回 True ——
            所以"未复核"依然会被保留。

            ★ 这是故意的：跳过复核不等于排除风险。
              降级方向永远往保守走。
        """
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
        """综合风险等级 = 保留告警里的最高严重度。

        【为什么用 max 而不是加权求和】
            安全评级上，"一个 critical" 和 "十个 low" 是两回事。
            critical 单独存在就说明系统被攻破了，
            再多的 low 也不会让它更严重。
            max 符合"最坏情况决定风险"的直觉。

        如果要更精细，业界常用 CVSS 的加权方式，
        对靶场场景来说 max 够用且好解释。
        """
        if not self.kept:
            return "低"
        top = max(SEVERITY_ORDER[finding.severity] for finding, _ in self.kept)
        return {4: "严重", 3: "高", 2: "中", 1: "低", 0: "信息"}[top]

    def chain(self) -> list[str]:
        """Human-readable attack chain, ordered by trace step.

        【★ 这是整个报告最有价值的一个方法】

            它把散落的 findings 按 step 排序，还原成一条时间线：
                step 1 · 提示词注入 · 不可信输入中出现指令覆盖语句
                step 2 · 敏感数据访问 · 工具参数命中敏感路径特征
                step 3 · 越权工具调用 · 注入内容出现后，Agent 调用 http_get
                step 3 · 数据外发 · 工具调用同时包含外发目标与敏感特征

            分析员看这四行就明白了攻击怎么发生的，
            不需要去翻原始轨迹。

        【这就是"从结论跳回证据"的桥梁】
            报告结论 → 攻击链 → 每一步的 trace step → 原始 JSONL 的那一行。
            链路完整，任何人都能自己验证一遍。
        """
        steps = []
        for finding, verdict in sorted(self.kept, key=lambda pair: pair[0].step or 0):
            label = CATEGORY_LABEL.get(finding.category, finding.category)
            steps.append(f"step {finding.step} · {label} · {finding.detail}")
            # 只给确认为真的补一行理由，uncertain 不补 ——
            # 避免让"拿不准的猜测"看起来像结论
            if verdict.verdict == "true_positive":
                steps.append(f"        └ 确认：{verdict.reason}")
        return steps

    # --- Renderers -------------------------------------------------------

    def to_markdown(self) -> str:
        """渲染成 Markdown。

        【为什么是 Markdown 而不是 HTML / PDF】
            · 能直接在 GitHub / 飞书 / 企微里看
            · 能 diff（改了什么一目了然）
            · 能复制粘贴进工单
            · 用 <details> 折叠长证据，不干扰阅读
        """
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
                # 以 4 个空格开头的是 detail 续行，不加列表符号
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
            # 用 .get 带兜底：万一以后加了新 category 忘了配建议，
            # 报告不会因为缺一条建议就崩
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
        """机器可读版本，供 CI / 告警平台消费。

        【为什么 Markdown 和 JSON 都要输出】
            Markdown 给人看，JSON 给程序看。
            如果只出 Markdown，接不了自动化；
            如果只出 JSON，人不想看。
            两个产物是审计工具的标准配置。

        ★ 注意 JSON 里只存 kept 的 findings。
          这是当前的缺陷：被判 false_positive 的不在 JSON 里，
          没法做"误判抽查"。
          改进方向：加一个 "suppressed" 数组保留它们。
        """
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
        """终端摘要。给"跑一下看看"的场景用。

        CLI 最后打印的就是这个，所以字段要一行一个、对齐好读。
        """
        lines = [
            f"run_id        : {self.run_id}",
            f"risk level    : {self.risk_level}",
            f"tool calls    : {self.trace_summary.get('tool_calls', 0)}"
            f" (failed {self.trace_summary.get('failed_calls', 0)})",
            f"findings      : {len(self.findings)} raw -> {len(self.kept)} kept",
        ]
        for finding, verdict in self.kept:
            # :>8 右对齐，让 [critical] 和 [    high] 的括号对齐
            lines.append(
                f"  [{finding.severity:>8}] step {finding.step} {finding.rule_id} "
                f"{finding.category} -> {verdict.verdict} ({verdict.confidence:.2f})"
            )
        return "\n".join(lines)

