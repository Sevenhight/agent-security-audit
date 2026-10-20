"""MCP-style tool registry.

Each tool is described by a JSON Schema, exactly like the ``tools/list``
payload an MCP server returns.  The agent loop picks tools through this
registry, and the sandbox layer wraps every call before it executes.

Keeping the schema and the implementation in one place means the agent
never sees a tool it cannot actually call, and every declared parameter is
validated before dispatch.
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
    """A callable tool plus the contract the model must respect."""

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
        _validate_arguments(self.input_schema, arguments)
        return self.handler(**arguments)


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")

    properties = schema.get("properties", {})
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    for key, value in arguments.items():
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            raise TypeError(f"argument {key} must be a string")
        if expected == "integer" and not isinstance(value, int):
            raise TypeError(f"argument {key} must be an integer")


class ToolRegistry:
    """Holds the tool set exposed to one agent run."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicated tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Compact tool list for the prompt."""
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
