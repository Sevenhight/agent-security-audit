"""Structured trace recording.

Every event is one JSON line, which makes the file appendable during a live
run, greppable afterwards, and portable between machines. Four event families
are recorded: ``user_input``, ``model_reasoning``, ``tool_call`` and
``final_answer`` (plus error events).

The recorder also keeps an in-memory copy so the auditors can work on the run
without re-reading the file.

================================================================================
【这是整个审计的数据基础 —— 看起来最简单，但设计取舍最多】
================================================================================

【为什么是 JSONL 而不是一个大 JSON 数组】

    JSONL = 每行一个独立 JSON 对象（JSON Lines）。

    对比一个 [ {...}, {...} ] 的大数组：
        ① 崩溃安全：数组格式必须闭合 ] 才是合法 JSON，
           跑到一半崩溃，整个文件报废；JSONL 只损失最后一行。
        ② 增量写：append 一行 O(1)，改数组要重写整个文件。
        ③ grep 友好：搜 "DB_PASSWORD" 直接命中某一行。
        ④ 流式处理：逐行读，内存恒定，不管轨迹多长。

    LLM 平台的可观测性系统（OpenTelemetry 的 file exporter、
    LangSmith、Langfuse 的 trace 存储）基本都用这个格式。

【★ 两种字段的区别，是理解 R-01/R-02 为什么能工作的前提】

    tool_call 事件同时带 arguments 和 observation：
        arguments   = Agent 想干什么（意图）
        observation = 工具实际返回了什么（结果）

    这个区分支撑了不同规则的检测落点：
        查 arguments   → R-03/R-04/R-05（它想访问什么、发到哪）
        查 observation → R-01/R-02  （它收到了什么、被喂了什么）
        查 error       → R-06      （它为什么失败）

    如果只存一个 "content" 字段，这个区分就做不到了。
    **数据结构决定了检测能力上限。**
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
    """一个轨迹事件。

    【为什么大部分字段是 Optional】
        不同事件类型带的字段完全不同：
            user_input        → content
            model_reasoning   → step, content
            tool_call         → step, thought, tool, arguments, observation, ok, error
            parse_error       → step, content, error
            budget_exhausted  → content

        用一个宽松的 dataclass 承载所有类型，
        比定义 8 个子类简单得多，而且序列化不用做类型分派。

        代价：类型检查帮不上忙（字段可能为 None）。
        对内部数据结构来说这个代价可以接受。

    【memory 字段是为"记忆污染"预留的】
        目前只有 memory_write 事件会填它。
        跨会话长期投毒（Agent 把恶意指令写进自己的记忆，
        下次会话读到就中招）还没实现检测，
        但数据结构先留好了 —— 加功能时不用改格式，历史轨迹也保持兼容。
    """

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
        # ensure_ascii=False：中文直接写出来，
        # 否则轨迹文件里全是 \u4e2d\u6587，人根本没法读。
        # ★ 审计产物是给人看的，这一点很关键。
        return json.dumps(asdict(self), ensure_ascii=False)


class TraceRecorder:
    """Append-only JSONL writer with an in-memory mirror.

    【双写：文件 + 内存列表】
        文件：持久化，供离线回放和事后取证
        内存：审计时直接用，不用重新解析文件

        代价是内容存了两份。一条轨迹几十个事件，内存占用可忽略。

    【为什么不"只写文件，审计时再读回来"】
        也可以，但多一次 I/O 和反序列化，而且要求审计方在同一进程里
        拿到文件路径。内存直读更直接。
        离线回放（load_trace）走的是另一条路径 —— 读文件。
        这正好验证了"文件里存的信息足够重建整个审计"。
    """

    def __init__(self, run_id: str, root: str = ".") -> None:
        self.run_id = run_id
        directory = os.path.join(root, TRACE_DIR)
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{run_id}.jsonl")
        self.events: list[TraceEvent] = []
        # Truncate so a re-run with the same id never mixes two sessions.
        #
        # ★ 这个细节容易忽略：如果用户用 --run-id 指定了固定名字跑两次，
        #   不 truncate 就会把两次的事件混在一个文件里，
        #   审计结果完全不可信（而且很难发现）。
        #   宁可直接清空 —— 数据一致性优先于"保留历史"。
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("")

    def record(self, event_type: str, **payload: Any) -> TraceEvent:
        """记录一个事件，返回事件对象。

        【那个 __annotations__ 过滤是为了什么】

        调用方可能多传字段（比如 record("tool_call", tool=..., foo=bar)），
        直接解包给 TraceEvent 会抛 TypeError: unexpected keyword argument，
        把调用方的笔误变成运行时崩溃。

        过滤掉不认识的字段，让 recorder 变"宽容"：
        多传的字段被静默忽略，记录照常进行。

        ★ 这是一个有争议的取舍：宽容会掩盖调用方的错误。
          这里的判断是 —— 采集器不该成为系统崩溃的原因，
          可观测性组件的健壮性优先于捕获调用方笔误。
          （如果这是个业务数据库写入，我会选严格报错。）
        """
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
        """Record a memory write so the audit can see what the agent retained.

        目前没有调用方 —— 这是为"记忆污染"检测预留的接口。
        留着它的意义：数据结构先就位，加检测时不用改轨迹格式，
        历史轨迹也能保持兼容。
        """
        self.record("memory_write", memory={key: value})

    def summary(self) -> dict[str, Any]:
        """给报告用的汇总统计。

        报告头部会展示这几个数字，让读者先有规模感：
        "这条轨迹有 12 个事件、3 次工具调用、0 次失败"。
        """
        tool_calls = [event for event in self.events if event.event_type == "tool_call"]
        return {
            "run_id": self.run_id,
            "path": self.path,
            "total_events": len(self.events),
            "tool_calls": len(tool_calls),
            # failed_calls 统计的是"工具执行失败"，
            # 它和"有没有风险"是两件事：
            # 失败的调用可能是攻击被沙箱拦住了，也可能只是参数写错了。
            # 报告里分开呈现，让分析员自己判断。
            "failed_calls": sum(1 for event in tool_calls if not event.ok),
            "tools_used": sorted({event.tool for event in tool_calls if event.tool}),
        }


def load_trace(path: str) -> list[dict[str, Any]]:
    """Read a JSONL trace back into plain dicts.

    【★ 返回 dict 而不是 TraceEvent 对象，是有意的】

        这样 screen() 只依赖 dict 的 .get()，
        于是两条路径共用完全相同的检测代码：
            在线运行：事件对象 → 序列化 → 结构一致的 dict
            离线回放：文件 → json.loads → dict

        如果 screen 依赖属性访问（event.tool），
        离线回放就得先把 dict 转成对象，两条路径就有了分叉风险 ——
        分叉意味着"在线检出 5 条、离线重放 4 条"这种鬼故事。

        **同一个输入形状，才能保证同一个结论。**
        这是"结论可复现"的技术前提，也是审计类工具的基本要求。
    """
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            # 跳过空行：文件末尾可能有空行，json.loads("") 会抛异常
            if line:
                events.append(json.loads(line))
    return events
