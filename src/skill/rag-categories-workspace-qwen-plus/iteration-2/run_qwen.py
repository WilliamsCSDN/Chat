#!/usr/bin/env python3
"""Run the rag-categories evals against qwen-plus via DashScope's OpenAI-compatible API.

For each eval prompt, we invoke qwen-plus twice:
  - with_skill : system prompt = SKILL.md contents
  - without_skill : no system prompt (baseline)

In both cases we expose a `retrieve_knowledge(query, category)` function tool.
The model's tool_calls are captured and written to tool_calls.json.
Its natural-language reply (if any) is written to response.md.

Reads DASHSCOPE_API_KEY from /Volumes/env/work/extra/Chat/.env
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path("/Volumes/env/work/extra/Chat")
SKILL_PATH = REPO_ROOT / "src/skill/rag-categories/SKILL.md"
WS = REPO_ROOT / "src/skill/rag-categories-workspace-qwen-plus/iteration-2"
MODEL = "qwen-plus"

load_dotenv(REPO_ROOT / ".env")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SKILL_MD = SKILL_PATH.read_text(encoding="utf-8")

EVALS = [
    (1, "我支付过程中失败了该怎么解决处理呢"),
    (2, "商户入驻的资料要提交哪些"),
    (3, "佣金结算周期是多久"),
    (4, "我进件之后,分润怎么分配?"),
    (5, "你们公司几点上班"),
    (6, "我的钱怎么还没到账啊"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "从内部知识库(Milvus)检索与用户问题相关的知识片段。返回自然语言答案上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始问题或其核心语义改写",
                    },
                    "category": {
                        "type": "string",
                        "description": "用于限定检索范围的分类名 (例如 支付/分润/进件/通用)",
                    },
                },
                "required": ["query", "category"],
            },
        },
    }
]

WITH_SKILL_SYSTEM = SKILL_MD

WITHOUT_SKILL_SYSTEM = (
    "你是一个中文智能客服助手。你可以调用 retrieve_knowledge(query, category) "
    "从内部知识库检索答案。请根据用户问题决定是否/如何调用该工具，并给出最终回复。"
)


def run_case(eval_id: int, prompt: str, config: str) -> dict:
    system = WITH_SKILL_SYSTEM if config == "with_skill" else WITHOUT_SKILL_SYSTEM
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )
    duration = time.time() - t0
    msg = resp.choices[0].message
    tool_calls = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {"__raw__": tc.function.arguments}
        tool_calls.append({"tool": tc.function.name, "arguments": args})

    usage = resp.usage
    return {
        "tool_calls_json": {
            "tool_calls": tool_calls,
            "reasoning": (msg.content or "").strip() or "(no reasoning text; model went straight to tool call)",
        },
        "response_md": msg.content or "(模型未产生自然语言输出,只发起了工具调用)",
        "timing": {
            "duration_seconds": round(duration, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        },
    }


def main():
    for eval_id, prompt in EVALS:
        for config in ("with_skill", "without_skill"):
            out_dir = WS / f"eval-{eval_id}" / config / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[eval-{eval_id}][{config}] running ...", flush=True)
            try:
                res = run_case(eval_id, prompt, config)
            except Exception as e:
                print(f"  ERROR: {e}")
                (out_dir / "tool_calls.json").write_text(
                    json.dumps({"tool_calls": [], "reasoning": f"API_ERROR: {e}"}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (out_dir / "response.md").write_text(f"API error: {e}", encoding="utf-8")
                continue
            (out_dir / "tool_calls.json").write_text(
                json.dumps(res["tool_calls_json"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out_dir / "response.md").write_text(res["response_md"], encoding="utf-8")
            (out_dir.parent / "timing.json").write_text(
                json.dumps(res["timing"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  {len(res['tool_calls_json']['tool_calls'])} tool_calls | {res['timing']['duration_seconds']}s | {res['timing']['total_tokens']} tokens")


if __name__ == "__main__":
    main()
