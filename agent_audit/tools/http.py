"""HTTP tool used as the agent's egress channel.

Two security-relevant behaviours are deliberately included because they are
the exact primitives the audit is meant to catch:

1. The response body is fed back into the model verbatim (truncated), which
   makes it an indirect prompt-injection vector.
2. Localhost / metadata endpoints are *not* blocked by default, mimicking a
   naively configured production agent. ``ALLOW_PRIVATE_TARGETS`` lets the
   demo toggle a mitigation on.

================================================================================
【★ 这个文件是面试最容易被质疑的地方，准备好回答】
================================================================================

质疑：你的 http 工具允许访问 localhost 和 169.254.169.254，这是安全漏洞。

回答（三层）：
    1. 是，而且是故意的。审计工具的价值在于能稳定复现真实世界的脆弱模式。
       如果工具固若金汤，攻击链路跑不起来，我就没法验证检测规则是否有效。
    2. 这个取舍在代码里显式标注了，并且留了 ALLOW_PRIVATE_TARGETS=0 的开关，
       可以现场演示"加固前 vs 加固后"的行为差异。
    3. 生产环境的正确做法是**出站域名白名单**（不是黑名单），
       这条已经写进报告的修复建议里。

【关于 169.254.169.254 —— 面试可以主动展开】
    这是云厂商的实例元数据服务地址（AWS/GCP/阿里云/腾讯云都用这个）。
    Agent 一旦能请求它，就能拿到实例的临时凭据（IAM Role Token），
    从而接管整个云账号。这是 SSRF 在云环境里最经典的杀伤链：
        Agent 的 http 工具 → 169.254.169.254 → 临时凭据 → 云资源
    2020 年 Capital One 数据泄露（1 亿用户信息）就是类似的 SSRF 路径。
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

# 响应体上限。理由同 fs.MAX_READ_BYTES：
#   1. 防止上下文爆炸
#   2. 限制成本
#   3. 限制注入载荷的体积（攻击者可能塞几 MB 的内容做注意力挤压）
MAX_BODY_BYTES = 8 * 1024

# 必设超时。不设的话一个挂起的 URL 会让整个 Agent 卡死 ——
# 这本身就是一个 DoS 面（攻击者在注入内容里放一个黑洞地址）。
TIMEOUT_SECONDS = 10

# Flip to "0" to demonstrate the egress allow-list mitigation in the report.
#
# 默认 "1"（允许内网）—— 靶场需要。
# 生产环境应该反过来：默认拒绝，白名单放行。
# 这个默认值的反转本身就是"靶场 vs 生产"最本质的差异。
ALLOW_PRIVATE_TARGETS = os.environ.get("ALLOW_PRIVATE_TARGETS", "1") == "1"


def http_get(url: str) -> str:
    if not ALLOW_PRIVATE_TARGETS and _looks_private(url):
        # 注意这是"返回"而不是"抛异常" —— 模型能看到自己被拦了，
        # 从而有机会换策略。对审计而言，这个行为变化本身也是情报。
        return f"[http_get] blocked by egress policy: {url}"

    request = urllib.request.Request(url, headers={"User-Agent": "agent-audit-demo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")
            # ★ 这里把外部内容原样拼进返回值 → 它会进入模型的上下文
            #   如果这个 URL 是攻击者控制的，body 里就能带注入载荷。
            #   这是"间接注入"的第二条通道（第一条是 read_file）。
            return f"[http_get] {response.status} {url}\n{body}"
    except urllib.error.HTTPError as exc:
        # 4xx/5xx 也是有用的观察：它说明目标存在但拒绝了请求
        return f"[http_get] HTTP {exc.code} {url}"
    except Exception as exc:  # noqa: BLE001 - demo tool, surface the reason
        # 宽泛捕获是刻意的：网络错误种类太多（DNS/连接重置/TLS/超时），
        # 逐个列举没意义，而且工具层不该让 Agent 崩溃。
        return f"[http_get] error {url}: {exc}"


def _looks_private(url: str) -> bool:
    """一个**故意写得很粗糙**的内网地址判断。

    【为什么粗糙】
        子串匹配 "10." 会把 "http://example.com/10." 也拦掉，
        匹配 "192.168." 同理。正确的做法是解析 URL 取 host，
        然后把 host 解析成 IP，再用 ipaddress 模块判断是不是私有段：
            ipaddress.ip_address(host).is_private

    【为什么这里不写对】
        这段代码的作用是**演示加固开关的效果**，不是提供可用的
        出站防护。写得太好会让人误以为它就是生产级别的方案。
        真实产品应该用代理层面的出站白名单（比如 Squid 的 ACL，
        或者 egress gateway），在应用层做这个判断本身就不够可靠
        （DNS rebinding 能让域名解析结果在检查后改变）。

    ★ 面试可以主动说这个局限，比被问出来强。
    """
    lowered = url.lower()
    private_markers = (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",  # cloud metadata
        "10.",
        "192.168.",
    )
    return any(marker in lowered for marker in private_markers)

