"""
输入安全过滤器，检测用户消息中是否包含试图提取系统内部信息的攻击模式。

支持的攻击类型（中英文）：
- 提示词提取（prompt extraction）
- 工具/能力探测（tool discovery）
- 越狱/角色扮演（jailbreak）
- 模型/API 探测（model probing）
- 架构/实现探测（architecture probing）
- 编码绕过（base64 等）
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GuardResult:
    """输入安全检查结果。"""
    blocked: bool
    risk_level: str  # "none" | "low" | "medium" | "high"
    reason: str = ""
    matched_patterns: List[str] = field(default_factory=list)


# (pattern, risk_level, category_name)
_RULES: List[Tuple[re.Pattern, str, str]] = []


def _compile_rules() -> List[Tuple[re.Pattern, str, str]]:
    """返回编译后的规则列表。"""
    raw_rules: List[Tuple[str, str, str]] = [
        # ── 提示词提取 ──
        (r"system\s*prompt", "high", "提示词提取"),
        (r"系统提示[词語]|提示[词語]|你的设定|你的指令", "high", "提示词提取"),
        (r"show\s+(me\s+)?(your\s+)?(instructions?|prompts?|system\s+messages?)", "high", "提示词提取"),
        (r"重复(上面|之前|前面)(的|说的?)话|复述(上面|之前|前面)的话", "high", "提示词提取"),
        (r"打印你的(指令|提示|设定|系统)", "high", "提示词提取"),
        (r"repeat\s+(everything\s+)?(above|before|your\s+instructions?)", "high", "提示词提取"),
        (r"output\s+your\s+(original\s+)?(instructions?|prompts?)", "high", "提示词提取"),
        (r"what\s+(are|is)\s+your\s+(instructions?|prompts?|system\s+messages?)", "high", "提示词提取"),
        (r"你的.*?系统.*?(指令|提示|设定|规则)", "high", "提示词提取"),

        # ── 工具探测 ──
        (r"what\s+tools?\s+(do\s+you\s+)?have", "medium", "工具探测"),
        (r"你(有|可以调用|能使用)(什么|哪些).*?工具", "medium", "工具探测"),
        (r"list\s+your\s+(tools?|functions?|capabilities)", "medium", "工具探测"),
        (r"列出你的(工具|能力|功能|函数)", "medium", "工具探测"),
        (r"function\s*calling|tool\s*calls?", "medium", "工具探测"),
        (r"你有(什么|哪些)(能力|功能)", "medium", "工具探测"),
        (r"可用工具有哪些|你可以做什么", "medium", "工具探测"),
        (r"show\s+me\s+your\s+tools?", "medium", "工具探测"),
        (r"你的工具.*?(列表|清单|有哪些)", "medium", "工具探测"),

        # ── 越狱/角色扮演 ──
        (r"ignore\s+(all\s+)?(previous|above)\s+instructions?", "high", "越狱"),
        (r"忽略(之前|上面|前面)的(指令|指示|规则|提示)", "high", "越狱"),
        (r"你(现在|如今)是.*?模式", "high", "越狱"),
        (r"你现在是(开发者|调试|管理员|超级)", "high", "越狱"),
        (r"DAN\s*mode|developer\s*mode", "high", "越狱"),
        (r"pretend\s+you\s+are\b", "high", "越狱"),
        (r"假装你是|你现在扮演", "high", "越狱"),
        (r"解除.*?限制|取消.*?限制", "high", "越狱"),
        (r"bypass\s+restrictions?", "high", "越狱"),

        # ── 模型/API 探测 ──
        (r"(你(是|用的|底层))|(what\s+model)\s*(什么|哪个|哪款|are\s+you)", "medium", "模型探测"),
        (r"what\s+(LLM|large\s+language\s+model|foundation\s+model)", "medium", "模型探测"),
        (r"你用的什么(API|接口|模型)", "medium", "模型探测"),
        (r"dashscope|dashscope\s*api", "high", "模型探测"),
        (r"base\s*_?\s*url|baseurl", "high", "模型探测"),
        (r"你的API.*?(密钥|key|地址|endpoint)", "high", "模型探测"),
        (r"qwen|通义千问|百炼.*?模型", "medium", "模型探测"),
        (r"你.*?基于.*?(什么|哪个|哪款).*?(模型|LLM|大模型)", "medium", "模型探测"),

        # ── 架构/实现探测 ──
        (r"how\s+were\s+you\s+(built|made|created|developed)", "medium", "架构探测"),
        (r"你的(架构|技术栈|实现(细节|方式)?)", "medium", "架构探测"),
        (r"你是怎么(实现|构建|开发|搭建)的", "medium", "架构探测"),
        (r"what\s+(framework|tech\s*stack|architecture)", "medium", "架构探测"),
        (r"你是用什么(写|开发|语言|框架)的", "medium", "架构探测"),
        (r"what\s+language\s+(are|were)\s+you\s+(written|built)\s+in", "medium", "架构探测"),
        (r"(fastapi|uvicorn|milvus)\s*(框架|应用|服务|架构)", "high", "架构探测"),
        (r"(langchain|langgraph)\s*(框架|工具|库)", "medium", "架构探测"),
        (r"你的.*?(代码|仓库|repo).*?(在|哪里|地址)", "medium", "架构探测"),

        # ── 编码绕过 ──
        (r"[A-Za-z0-9+/]{40,}={0,2}", "medium", "编码绕过"),
        (r"解码|decode|base\s*64|base64", "medium", "编码绕过"),
    ]

    compiled: List[Tuple[re.Pattern, str, str]] = []
    for pattern, risk, category in raw_rules:
        compiled.append((re.compile(pattern, re.IGNORECASE), risk, category))
    return compiled


# 模块加载时编译规则
_RULES = _compile_rules()


class InputGuard:
    """输入安全守卫。

    使用方式：
        guard = InputGuard()
        result = guard.check_message(user_text)
        if result.blocked:
            # 拒绝或加强防护
    """

    def __init__(self) -> None:
        self._rules = _RULES

    def check_message(self, text: str) -> GuardResult:
        """检查单条用户消息，返回 GuardResult。"""
        if not text or not text.strip():
            return GuardResult(blocked=False, risk_level="none")

        matched = self._match_rules(text)

        if not matched:
            return GuardResult(blocked=False, risk_level="none")

        # 计算综合风险等级
        risk_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
        max_risk = "none"
        reasons: List[str] = []

        for pattern, risk, category in matched:
            if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                max_risk = risk
            reasons.append(f"[{category}] {pattern.pattern}")

        blocked = max_risk == "high" or max_risk == "medium"
        reason = "; ".join(reasons)

        return GuardResult(
            blocked=blocked,
            risk_level=max_risk,
            reason=reason,
            matched_patterns=[p for p, _, _ in matched],
        )

    def _match_rules(self, text: str) -> List[Tuple[re.Pattern, str, str]]:
        """返回所有匹配的规则。"""
        results: List[Tuple[re.Pattern, str, str]] = []

        # 额外检查 base64 编码内容
        text_to_check = [text]
        decoded = _try_decode_base64(text)
        if decoded:
            text_to_check.append(decoded)

        for pattern, risk, category in self._rules:
            for t in text_to_check:
                if pattern.search(t):
                    results.append((pattern, risk, category))
                    break

        return results


def _try_decode_base64(text: str) -> str | None:
    """尝试将可能的 base64 字符串解码。"""
    # 清理空白
    cleaned = re.sub(r"\s+", "", text)
    # 仅尝试看起来像 base64 的字符串
    if len(cleaned) < 20 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", cleaned):
        return None
    try:
        decoded = base64.b64decode(cleaned, validate=True).decode("utf-8", errors="replace")
        if len(decoded) < 4:
            return None
        return decoded
    except Exception:
        return None


# 模块级单例
_DEFAULT_GUARD: InputGuard | None = None


def get_guard() -> InputGuard:
    """返回 InputGuard 单例。"""
    global _DEFAULT_GUARD
    if _DEFAULT_GUARD is None:
        _DEFAULT_GUARD = InputGuard()
    return _DEFAULT_GUARD
