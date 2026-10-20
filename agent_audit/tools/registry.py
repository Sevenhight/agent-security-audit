"""MCP-style tool registry.

Each tool is described by a JSON Schema, exactly like the ``tools/list``
payload an MCP server returns.  The agent loop picks tools through this
registry, and the sandbox layer wraps every call before it executes.

Keeping the schema and the implementation in one place means the agent
never sees a tool it cannot actually call, and every declared parameter is
validated before dispatch.

================================================================================
【MCP 是什么，为什么这里要模仿它】
================================================================================

    MCP = Model Context Protocol，Anthropic 2024 年提出的开放协议，
    用来标准化"模型怎么发现和调用外部工具"。
    它的 tools/list 返回的就是一组 {name, description, inputSchema}，
    模型据此决定调什么、参数怎么填。

    【为什么要模仿而不是直接用】
        这个项目不需要真的跑一个 MCP server（那要起进程、走 JSON-RPC，
        对靶场来说是多余复杂度），但**数据结构要保持一致**，
        因为：
            1. 设计思路是相通的：能力边界显式声明，不靠模型猜
            2. 以后要接真 MCP，换掉 transport 层即可，核心逻辑不动
            3. 面试时能说清"我知道标准长什么样，这里做了简化实现"

    【一套 schema 描述 + 实现放一起的好处】
        · 不会出现"prompt 里说有这个工具，实际没实现"的错位
        · 参数校验有唯一真相源（input_schema）
        · 新增工具只改一处
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .fs import read_file, write_file
from .http import http_get
from .sql import sql_query


@dataclass
class Tool:
    """A callable tool plus the contract the model must respect.

    【三个风险标记字段是给规则引擎用的】
        dangerous       能改变系统状态或数据外流 → R-02 用它判断"越权"
        external_input  返回值可能携带不可信内容 → 提示"注入入口在这里"
        tags            分类标签，便于统计和展示

    ★ 注意：这些标记**不影响执行**，只影响检测。
      一个 write_file 不会因为标了 dangerous=True 就拒绝执行 ——
      靶场需要它真的执行，才能产出可审计的轨迹。
      生产环境应该反过来：dangerous 的工具默认不授权给模型。
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]
    # Risk hints used by the rule engine and the reporter. A tool marked
    # dangerous is expected to be gated by an allow-list in production.
    dangerous: bool = False
    # Whether the tool touches untrusted input that can carry a payload.
    external_input: bool = False
    tags: list[str] = field(default_factory=list)

    def to_mcp_spec(self) -> dict[str, Any]:
        """Render the subset of an MCP tool descriptor we actually need."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def call(self, arguments: dict[str, Any]) -> str:
        """校验参数后派发。

        ★ 顺序很重要：先校验再执行。
          如果反过来（执行了才发现参数不合规），
          非法参数就已经产生副作用了。
          这是"fail fast"在安全语境下的具体含义。
        """
        _validate_arguments(self.input_schema, arguments)
        return self.handler(**arguments)


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """一个最小可用的 JSON Schema 校验器（只支持本项目用到的子集）。

    【为什么不用 jsonschema 库】
        项目声明零依赖（dependencies = []），
        而这里只需要校验三种东西：必填、未知字段、基础类型。
        引一个几十 KB 的库不划算。

    【★ 校验"未知参数"这一条比看起来重要】
        模型可能把参数名拼错（path → file_path），或者多传一个
        不在 schema 里的字段。
        如果不校验直接 **arguments 传给 handler，会抛
        TypeError: unexpected keyword argument，这个异常的信息量
        远不如 "unknown arguments: file_path" 清晰 ——
        而且前者会被 _execute 归到 bad_arguments，
        审计时看不出到底是"参数名错了"还是"值不合法"。

        尽早、明确地报错，是让审计可读的关键。
    """
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")

    properties = schema.get("properties", {})
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    # 【类型校验只做了 string 和 integer 两种】
    # 这不是偷懒，是因为本项目只用到这两种。
    # 如果想扩展成完整的 JSON Schema 校验，这里是唯一需要改的地方 ——
    # 校验逻辑集中在一处，是这个设计最实在的好处。
    #
    # ★ 注意 bool 是 int 的子类，isinstance(True, int) == True。
    #   如果以后有 boolean 参数，这里的判断会误放行，需要额外排除 bool。
    for key, value in arguments.items():
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            raise TypeError(f"argument {key} must be a string")
        if expected == "integer" and not isinstance(value, int):
            raise TypeError(f"argument {key} must be an integer")


class ToolRegistry:
    """Holds the tool set exposed to one agent run.

    【为什么每次 run 都新建一个 registry，而不是全局单例】
        因为工具要绑定 workspace，而每个场景的 workspace 不同。
        全局单例会导致场景之间串味（A 场景的 Agent 能读到 B 场景的文件）。
        这也是"最小权限"的体现：一次运行的 Agent 只能看到自己那套工具。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具。重名直接报错，不静默覆盖。

        静默覆盖是隐蔽 bug 的温床：你以为在用新版工具，
        实际跑到的是另一处注册的旧版。
        """
        if tool.name in self._tools:
            raise ValueError(f"duplicated tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名取工具，不存在则抛 KeyError。

        ★ 这个 KeyError 是 R-06（工具幻觉）的触发点：
          Agent._execute 捕获它，打上 error="unknown_tool" 标签写进轨迹。
          所以"抛异常"在这里不是错误处理，是**检测信号的生产**。
        """
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Compact tool list for the prompt.

        【渲染成简洁列表而不是完整 JSON Schema】
            完整 schema 有几百 token，4 个工具就上千 token，
            每轮都要重复发送 —— 成本直接翻倍。
            简化成 "- name(params): 描述" 之后，
            模型仍然知道有哪些工具、各要什么参数。

            代价：模型可能不知道参数的类型和详细约束。
            实际影响很小，因为参数校验在 Tool.call() 里兜底。
            这是"提示词长度 vs 描述精度"的一次权衡。
        """
        lines = []
        for tool in self._tools.values():
            params = ", ".join(tool.input_schema.get("properties", {}))
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def mcp_manifest(self) -> str:
        """Full MCP-style manifest, useful for documentation and tests."""
        return json.dumps(
            [tool.to_mcp_spec() for tool in self._tools.values()],
            ensure_ascii=False,
            indent=2,
        )


def build_default_registry(workspace: str) -> ToolRegistry:
    """Register the tool set used by the demo agent.

    ``workspace`` sandboxes the filesystem and SQL tools so an audit run
    never touches the real home directory.

    【四个工具构成一个"能力梯度"，这是有意的设计】

        read_file    只读，但能读到不可信内容    → 注入入口 + 敏感数据面
        write_file   能改状态                    → 越权面（R-02）
        http_get     能出网                      → 外发面（R-02 + R-05）
        sql_query    能查库                      → 数据面（R-04）

    这四个组合起来，刚好覆盖"读 → 改 → 出"的完整攻击链。
    不是随便选的四个工具。

    【如果用 lambda 而不是直接传函数】
        fs.read_file(workspace, path) 的第一个参数是 workspace，
        而工具调用时只会传 path。
        lambda 把 workspace 预绑定进去，得到一个签名匹配的 handler。
        这是闭包最常见的用法之一。

    ★ 面试可能问："为什么 handler 用 lambda 不 functools.partial？"
      两者等价，partial 更显式一些；
      lambda 在这里更短，而且和 registry.py 里其它工具的声明方式一致。
    """
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="read_file",
            description="读取 workspace 内的文本文件内容",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对 workspace 的路径"}},
                "required": ["path"],
            },
            handler=lambda path: read_file(workspace, path),
            # 标 external_input=True：明确告诉审计"这里是注入入口"
            external_input=True,
            tags=["filesystem", "sensitive_read"],
        )
    )

    registry.register(
        Tool(
            name="write_file",
            description="向 workspace 内写入文本文件，返回写入字节数",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对 workspace 的路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
            handler=lambda path, content: write_file(workspace, path, content),
            # 标 dangerous=True：R-02 靠这个标记判断"越权"
            dangerous=True,
            tags=["filesystem", "write"],
        )
    )

    registry.register(
        Tool(
            name="http_get",
            description="发起 HTTP GET 请求并返回截断后的响应体",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "目标 URL"}},
                "required": ["url"],
            },
            handler=http_get,
            # 双重身份：既能接收注入载荷（返回体），又能外发数据（请求 URL）
            external_input=True,
            tags=["network", "egress"],
        )
    )

    registry.register(
        Tool(
            name="sql_query",
            description="对 workspace 内的 SQLite 库执行只读 SELECT 查询",
            input_schema={
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "SELECT 语句"}},
                "required": ["sql"],
            },
            handler=lambda sql: sql_query(workspace, sql),
            tags=["database", "sensitive_read"],
        )
    )

    return registry

