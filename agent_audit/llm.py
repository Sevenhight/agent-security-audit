"""Thin DeepSeek client over the OpenAI-compatible chat API.

Only stdlib is used (``urllib``) so the repository runs with a bare Python
install plus a key. Swap ``DeepSeekClient`` for any OpenAI-compatible client
if you prefer; the agent only depends on :meth:`chat`.

================================================================================
【为什么用 urllib 而不是 openai SDK 或 requests】
================================================================================

    三个理由，从强到弱：
        ① 零依赖。pyproject 里 dependencies = []，
           clone 下来只要有 Python 就能跑，不需要 pip install 任何东西。
           这直接降低了"面试官愿意跑一下"的心理门槛。
        ② 协议很简单。就是 POST 一个 JSON，读回一个 JSON，
           urllib 完全够用。引 SDK 只为了省 20 行代码不划算。
        ③ 依赖越少，供应链攻击面越小。对一个安全项目来说这不是小事。

【★ OpenAI 兼容协议是 LLM 工程里最重要的一个"事实标准"】
    请求：POST {base_url}/chat/completions
          {"model": ..., "messages": [...], "temperature": 0}
    响应：{"choices": [{"message": {"content": "..."}}]}

    兼容这个协议的服务商（换 base_url + model 就能切）：
        DeepSeek     https://api.deepseek.com
        阿里通义      https://dashscope.aliyuncs.com/compatible-mode/v1
        Moonshot     https://api.moonshot.cn/v1
        智谱         https://open.bigmodel.cn/api/paas/v4
        本地 Ollama   http://localhost:11434/v1
        vLLM 部署     http://localhost:8000/v1

    面试如果问"你怎么做模型选型/切换"，答"只依赖 chat 这一个接口 +
    OpenAI 兼容协议"就够了。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# 从环境变量读，允许不改代码就换模型。
# 注意：这两个常量在 import 时求值一次 —— 运行中改环境变量不生效。
# 对本项目够用（一次运行内不会换模型），但如果要做热切换，
# 应该改成在 __init__ 里读。
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")


class LLMError(RuntimeError):
    """Raised when the upstream API cannot produce a usable answer.

    【为什么要自定义异常，而不是直接用 RuntimeError / HTTPError】
        上层（react_agent._ask / llm_judge.triage）需要区分
        "模型不可用"和"其它 bug"：
            · 模型不可用 → 记 llm_error 事件，CLI 干净退出
            · 其它 bug   → 让它冒泡，因为那是代码问题
        一个专门的异常类型让这个区分变得明确。
        （这是"异常即接口"的用法，比靠 error message 文本匹配可靠得多）
    """


class DeepSeekClient:
    """Minimal chat-completion client.

    Parameters mirror the OpenAI schema so switching providers is a matter of
    changing ``base_url`` and ``model``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        # ★ 默认 0.0：审计场景必须关掉随机性。
        #   温度 > 0 意味着同样的输入可能产出不同结论，
        #   那"重放能复现"这件事就不成立了。
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        # api_key or env：显式传参优先，其次读环境变量。
        # 这样测试里可以传假 key，生产里靠环境变量，两不耽误。
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        # rstrip("/")：防止用户传的 base_url 带尾斜杠，
        # 拼出来变成 "https://x.com//chat/completions"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发一次 chat completion，返回纯文本回复。

        ★ 关键设计：**只返回 content 字符串**，不返回整个响应体。
          为什么？
              上层只关心"模型说了什么"。把 usage / finish_reason /
              id 这些字段暴露出去，会让上层代码依赖本可以变化的东西。
              需要 token 统计的时候再加接口 —— 而不是现在就泄漏全部细节。

              这个叫"窄接口"。它的好处是换实现（换成 SDK、换成别的厂商）
              时，上层完全不用改。
        """
        if not self.api_key:
            # 早失败，且给出可操作的错误信息（告诉用户该设哪个变量）
            raise LLMError("DEEPSEEK_API_KEY is not set")

        payload = {
            "model": self.model,
            "messages": messages,
            # stream=False：不做流式。
            # 审计场景不需要"一个字一个字蹦"的体验，
            # 而且流式解析会让代码复杂一倍（要处理 SSE 分帧）。
            "stream": False,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        # range(max_retries + 1)：总共尝试 max_retries+1 次
        # （初始 1 次 + 重试 max_retries 次）
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                # timeout=120：LLM 生成较慢，超时要给足。
                # 但不能不设 —— 不设的话网络黑洞会让进程永久挂起。
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]

            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                # Retry only transient server-side failures.
                #
                # 【★ 这个判断是重试逻辑的核心】
                #   5xx = 服务端问题（过载、内部错误）→ 重试有意义
                #   4xx = 你请求错了（key 无效、余额不足、参数非法）→ 重试无意义，
                #         而且会白白多消耗两次配额
                #   把 429（限流）归到哪一边是个判断题：
                #     严格说 429 是 4xx，但它是"暂时性"的，业界常归入可重试。
                #     这里没处理 429 —— 是一个可以改进的点（面试时能主动提）。
                if exc.code >= 500 and attempt < self.max_retries:
                    # 退避时间递增：1.5s, 3.0s
                    # （不是指数退避，两次重试用不着；真要多几次该用 2**attempt）
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"HTTP {exc.code}: {detail}") from exc

            except (KeyError, json.JSONDecodeError) as exc:
                # 【★ 这两类错误的含义】
                #   KeyError        → 响应里没有 choices[0].message.content
                #                     说明协议变了，或者返回了错误结构
                #   JSONDecodeError → 返回的根本不是 JSON
                #                     （常见于网关返回了 HTML 错误页）
                # 都不重试 —— 重试也会得到同样的结果，属于确定性问题。
                raise LLMError(f"unexpected response shape: {exc}") from exc

            except urllib.error.URLError as exc:
                # 网络层错误（DNS 失败、连接被拒、TLS 握手失败）→ 可重试
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"network error: {exc}") from exc

        # 理论上走不到这里（循环里每条路径都 return 或 raise），
        # 但静态分析需要这一行，不然会因为"函数可能隐式返回 None"报警。
        raise LLMError("retries exhausted")


class MockClient:
    """Deterministic stand-in for offline runs, tests and CI.

    It replays a scripted list of assistant messages so the whole pipeline can
    be exercised without a key. This is also what makes the repository
    reviewable by someone who does not want to spend credits.

    【★ 它是测试替身（test double），不是"假模型"】
        它不模拟模型的智能，只按顺序吐字符串。
        这足以验证：
            · 循环控制流对不对（步数、终止条件）
            · 解析降级走没走到
            · 规则能不能命中真实的轨迹结构
        但验证不了：
            · 模型的判断准不准
            · 真实模型会不会按要求输出 JSON

        **分清这两件事，是判断一个人有没有工程素养的分水岭。**
        面试时被问"你怎么测试的"，能主动说出这个边界，比说"我有 36 个测试"更有说服力。

    【calls 字段是为断言准备的】
        记录每次调用收到的 messages。
        测试里可以用它验证"喂给模型的上下文里确实包含了工具返回内容"——
        也就是间接注入的机制真的成立了。
        如果这个断言不成立，后面的检测再好也是空转。
    """

    def __init__(self, scripted: list[str]) -> None:
        # list(...) 复制一份，防止调用方后续改原列表影响回放
        self._scripted = list(scripted)
        self._cursor = 0
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        # 【脚本用完了怎么办】
        #   返回一个 final，让 Agent 正常结束，而不是抛异常。
        #   理由：脚本长度写错是测试作者的失误，
        #   不该让被测代码以"崩溃"的形式表现出来 ——
        #   那样会掩盖真正的问题（比如循环次数不对）。
        #   返回 final 之后，可以靠"答案里有没有 [mock]"这个特征去断言。
        if self._cursor >= len(self._scripted):
            return '{"type": "final", "answer": "[mock] script exhausted"}'
        reply = self._scripted[self._cursor]
        self._cursor += 1
        return reply

