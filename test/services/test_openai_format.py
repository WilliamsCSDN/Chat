"""openai_format 规范化单测。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.services.openai_format import (
    build_openai_request_body,
    chunk_to_openai_delta,
    finish_reason_from_message,
    lc_message_to_openai,
)


class OpenAIFormatTests(unittest.TestCase):
    def test_human_message(self):
        msg = lc_message_to_openai(HumanMessage(content="你好"))
        self.assertEqual(msg, {"role": "user", "content": "你好"})

    def test_assistant_with_tool_calls(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_wether", "args": {"city": "北京"}}],
        )
        msg = lc_message_to_openai(ai)
        self.assertEqual(msg["role"], "assistant")
        self.assertIsNone(msg["content"])
        self.assertEqual(
            msg["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_wether",
                        "arguments": '{"city": "北京"}',
                    },
                }
            ],
        )
        self.assertEqual(finish_reason_from_message(ai), "tool_calls")

    def test_tool_message(self):
        msg = lc_message_to_openai(
            ToolMessage(content="晴 25C", tool_call_id="call_1", name="get_wether")
        )
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_1")
        self.assertEqual(msg["content"], "晴 25C")

    def test_chunk_delta_content(self):
        class _Chunk:
            content = "hello"
            tool_call_chunks = []
            tool_calls = []

        delta = chunk_to_openai_delta(_Chunk())
        self.assertEqual(delta, {"content": "hello"})

    def test_build_request_body(self):
        body = build_openai_request_body(
            model="qwen-flash",
            messages=[HumanMessage(content="hi")],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        self.assertEqual(body["model"], "qwen-flash")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(len(body["tools"]), 1)

    def test_build_chat_completion(self):
        from src.services.openai_format import build_openai_chat_completion

        resp = build_openai_chat_completion(
            message={"role": "assistant", "content": "hi"},
            finish_reason="stop",
            model="qwen-flash",
            completion_id="chatcmpl-test",
            created=1710000000,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        self.assertEqual(resp["object"], "chat.completion")
        self.assertEqual(resp["id"], "chatcmpl-test")
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")
        self.assertEqual(resp["choices"][0]["message"]["content"], "hi")
        self.assertEqual(resp["usage"]["total_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
