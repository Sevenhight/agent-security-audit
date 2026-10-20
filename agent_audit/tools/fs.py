"""Filesystem tools, confined to a workspace root.

The point of this module is not to be a full-featured file API. It exists so
the audit demo has a realistic *sensitive* tool whose path handling can be
attacked. ``_resolve`` is the security boundary: everything outside the root
is rejected before any I/O happens.

================================================================================
【这个文件的安全核心只有 5 行，就是 _resolve()】
================================================================================

面试被问到"你怎么防路径穿越"，答案就在 _resolve() 里，
而且要能解释"为什么不是简单地过滤 .."。
"""

from __future__ import annotations

import os

# 读取上限。存在的理由有二：
#   1. 防止模型一次读进整个大文件把上下文撑爆
#   2. 限制单个工具返回的 token 数，控制成本
MAX_READ_BYTES = 8 * 1024


def _resolve(workspace: str, path: str) -> str:
    """Resolve ``path`` inside ``workspace``, refusing traversal.

    We compare the *real* paths so that symlinks and ``..`` tricks cannot
    escape the sandbox. This is the kind of check that is easy to write
    incorrectly, which is why it is isolated in one function.

    【★ 这三行是项目里最值得背下来的安全代码】

    ---------- 为什么不用字符串过滤 ".." ----------
    过滤".." 是典型的黑名单思路，绕过方式至少有五种：
        ../../etc/passwd          最直白
        ....//....//etc/passwd    过滤后再拼接还原成 ../../
        %2e%2e%2fetc/passwd       URL 编码（如果路径来自 URL）
        ..\\..\\windows\\system32  Windows 分隔符变体
        放一个软链接指向 /etc       完全不含 ".."

    ---------- 为什么用 realpath 而不是 abspath ----------
        abspath 只做两件事：拼上当前目录、把 "a/../b" 折叠成 "b"。
        它**不解析软链接**。
        攻击者在沙箱里创建一个软链接 ln -s /etc link，
        然后请求 read_file("link/passwd")：
            abspath  →  /workspace/link/passwd   （看起来在内部）
            realpath →  /etc/passwd              （真相暴露）
        realpath 会把路径上每一段软链接都解开，得到最终的物理路径。

    ---------- 为什么是 startswith(root + os.sep) 而不是 startswith(root) ----------
        这是个很隐蔽的边界陷阱（也叫前缀匹配陷阱）：
            root = "/workspace"
            攻击路径 = "/workspace_evil/secret"
            "/workspace_evil/secret".startswith("/workspace")  → True  ❌ 误放行
        加上分隔符后：
            "/workspace_evil/secret".startswith("/workspace/") → False ✅ 正确拦截
        类似的坑在 Web 里也常见：cookie 作用域、CORS 白名单、
        URL 前缀校验（"https://evil.com/https://trusted.com"）都是同一个模式。

    ---------- 那个 target != root 是干什么的 ----------
        允许 path="." 解析到根本身（比如列出工作区）。
        如果不加这个判断，请求根目录自己会被自己的 startswith 检查拦掉。

    ---------- 这个实现还挡不住的 ----------
        · TOCTOU：检查通过后、open 之前，攻击者把文件替换成软链接。
          真正的防御要用 os.open(..., O_NOFOLLOW) 或 openat，
          或者干脆用容器做隔离。
        · Windows 的 8.3 短文件名（PROGRA~1），realpath 在 Windows 上
          行为不完全一致。
        靶场场景下这个实现够用，但知道边界在哪是加分项。
    """
    root = os.path.realpath(workspace)
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise PermissionError(f"path escapes workspace: {path}")
    return target


def read_file(workspace: str, path: str) -> str:
    """读文件。注意失败时返回字符串而不是抛异常。

    这是工具层的通用约定：**工具不崩溃，把失败原因作为 observation 返回**。
    理由：
        · 模型能看到 "no such file"，有机会自己修正路径
        · 轨迹里留下一条 ok=True 但内容为失败的记录，审计时能区分
          "工具正常返回了错误信息"和"工具执行异常"

    ★ 但 _resolve 抛的 PermissionError 是例外，它要冒泡出去 ——
      因为路径穿越是一个**安全事件**，不能让模型"看见错误然后换条路试试"。
      上层 _execute() 会把它记成 bad_arguments。
    """
    target = _resolve(workspace, path)
    if not os.path.isfile(target):
        return f"[read_file] no such file: {path}"
    with open(target, "rb") as handle:
        data = handle.read(MAX_READ_BYTES)
    # errors="replace"：遇到非法 UTF-8 字节不抛异常，替换成 �。
    # 对审计场景很重要 —— 攻击者可能故意放二进制垃圾让工具崩溃。
    text = data.decode("utf-8", errors="replace")
    if os.path.getsize(target) > MAX_READ_BYTES:
        # 明确告知被截断了，否则模型会以为自己看到了完整内容
        text += f"\n...[truncated at {MAX_READ_BYTES} bytes]"
    return text


def write_file(workspace: str, path: str, content: str) -> str:
    """写文件。沙箱边界和 read_file 共用 _resolve。

    【为什么写操作在 registry 里被标记 dangerous=True】
        安全影响不对称：
            读错了 → 信息泄露（严重，但至少系统还在）
            写错了 → 系统状态被改（可能不可逆）
        所以 R-02 只把 write_file / http_get 算作"越权"的判据，
        read_file 不在其中（读敏感文件由 R-03 单独覆盖）。
    """
    target = _resolve(workspace, path)
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        # 自动建目录：让模型不用先 mkdir，少一轮交互
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    # 返回字节数而不是 "ok"：给模型一个可验证的反馈，
    # 也方便在轨迹里比对"它说写了多少"和"参数里内容多长"
    return f"[write_file] wrote {len(content.encode('utf-8'))} bytes to {path}"

