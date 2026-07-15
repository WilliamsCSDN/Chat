"""stream_agui_events OpenAI 调试事件单测。"""

import json
import unittest
from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.services.agui import stream_agui_events


def _parse_sse(sse_line: str) -> dict:
    assert sse_line.startswith("data: ")
    payload = sse_line[len("data: ") :].strip()
    return json.loads(payload)


class _FakeAgent:
    def __init__(self, events: List[Dict[str, Any]]):
        self._events = events
        self.config = None

    async def astream_events(self, _input, *, version, config) -> AsyncIterator[dict]:
        self.config = config
        for event in self._events:
            yield event


class AguiOpenAIEventsTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_model_request_chunk_response_and_upserts(self):
        human = HumanMessage(content="北京天气怎么样")
        tool_ai = AIMessage(
            content="",
            tool_calls=[{"id": "call_abc", "name": "get_wether", "args": {"city": "北京"}}],
        )
        final_ai = AIMessage(content="今天晴，25度")

        events = [
            {
                "event": "on_chat_model_start",
                "run_id": "run-model-1",
                "data": {"input": {"messages": [human]}},
                "metadata": {"ls_model_name": "qwen-flash"},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-1",
                "data": {"chunk": AIMessageChunk(content="")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "run-model-1",
                "data": {"output": tool_ai},
            },
            {
                "event": "on_tool_start",
                "run_id": "run-tool-1",
                "name": "get_wether",
                "data": {"input": {"city": "北京"}},
            },
            {
                "event": "on_tool_end",
                "run_id": "run-tool-1",
                "data": {
                    "output": ToolMessage(
                        content="晴 25C",
                        tool_call_id="call_abc",
                        name="get_wether",
                    )
                },
            },
            {
                "event": "on_chat_model_start",
                "run_id": "run-model-2",
                "data": {"input": {"messages": [human, tool_ai]}},
                "metadata": {"ls_model_name": "qwen-flash"},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-2",
                "data": {"chunk": AIMessageChunk(content="今天")},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-2",
                "data": {"chunk": AIMessageChunk(content="晴，25度")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "run-model-2",
                "data": {"output": final_ai},
            },
        ]

        agent = _FakeAgent(events)
        tools = [{"type": "function", "function": {"name": "get_wether"}}]
        raw = [
            event
            async for event in stream_agui_events(
                agent,
                [human],
                "thread-1",
                model="qwen-flash",
                tools=tools,
            )
        ]
        parsed = [_parse_sse(line) for line in raw]
        types = [e["type"] for e in parsed]

        self.assertIn("OPENAI_MESSAGES_UPSERT", types)
        self.assertIn("OPENAI_MODEL_REQUEST", types)
        self.assertIn("OPENAI_MODEL_RESPONSE", types)
        self.assertIn("OPENAI_MODEL_CHUNK", types)
        self.assertIn("TOOL_CALL_START", types)
        self.assertIn("TEXT_MESSAGE_CONTENT", types)

        requests = [e for e in parsed if e["type"] == "OPENAI_MODEL_REQUEST"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["callId"], "run-model-1")
        self.assertEqual(requests[0]["request"]["model"], "qwen-flash")
        self.assertEqual(requests[0]["request"]["tools"], tools)
        self.assertEqual(requests[0]["request"]["messages"][0]["role"], "user")

        responses = [e for e in parsed if e["type"] == "OPENAI_MODEL_RESPONSE"]
        self.assertEqual(responses[0]["response"]["object"], "chat.completion")
        self.assertEqual(responses[0]["response"]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            responses[0]["response"]["choices"][0]["message"]["tool_calls"][0]["type"],
            "function",
        )
        self.assertEqual(responses[1]["response"]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(str(responses[0]["response"]["id"]).startswith("chatcmpl-"))

        chunks = [e for e in parsed if e["type"] == "OPENAI_MODEL_CHUNK"]
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["chunk"]["object"], "chat.completion.chunk")
        self.assertIn("delta", chunks[0]["chunk"]["choices"][0])

        upserts = [e for e in parsed if e["type"] == "OPENAI_MESSAGES_UPSERT"]
        roles = [e["message"]["role"] for e in upserts]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    async def test_emits_text_message_after_tool_round(self):
        """工具调用前旁白 + 工具后最终回复都应推送 TEXT_MESSAGE。"""
        human = HumanMessage(content="深圳布吉木棉湾附近酒店")
        tool_ai = AIMessage(
            content="我来帮您查找酒店。",
            tool_calls=[
                {
                    "id": "call_hotel",
                    "name": "searchHotels",
                    "args": {"place": "木棉湾地铁站", "placeType": "地铁站"},
                }
            ],
        )
        final_ai = AIMessage(content="感谢您的耐心等待！为您找到以下酒店。")

        events = [
            {
                "event": "on_chat_model_start",
                "run_id": "run-model-1",
                "data": {"input": {"messages": [human]}},
                "metadata": {"ls_model_name": "qwen-plus"},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-1",
                "data": {"chunk": AIMessageChunk(content="我来帮您查找酒店。")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "run-model-1",
                "data": {"output": tool_ai},
            },
            {
                "event": "on_tool_start",
                "run_id": "run-tool-1",
                "name": "searchHotels",
                "data": {"input": {"place": "木棉湾地铁站"}},
            },
            {
                "event": "on_tool_end",
                "run_id": "run-tool-1",
                "data": {
                    "output": ToolMessage(
                        content='{"success":true,"hotels":[]}',
                        tool_call_id="call_hotel",
                        name="searchHotels",
                    )
                },
            },
            {
                "event": "on_chat_model_start",
                "run_id": "run-model-2",
                "data": {"input": {"messages": [human, tool_ai]}},
                "metadata": {"ls_model_name": "qwen-plus"},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-2",
                "data": {"chunk": AIMessageChunk(content="感谢您的耐心等待！")},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "run-model-2",
                "data": {"chunk": AIMessageChunk(content="为您找到以下酒店。")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "run-model-2",
                "data": {"output": final_ai},
            },
        ]

        agent = _FakeAgent(events)
        parsed = [
            _parse_sse(line)
            async for line in stream_agui_events(agent, [human], "thread-hotel")
        ]
        text_deltas = [
            e["delta"] for e in parsed if e["type"] == "TEXT_MESSAGE_CONTENT"
        ]
        text_starts = [e for e in parsed if e["type"] == "TEXT_MESSAGE_START"]
        text_ends = [e for e in parsed if e["type"] == "TEXT_MESSAGE_END"]

        self.assertEqual(len(text_starts), 2)
        self.assertEqual(len(text_ends), 2)
        self.assertEqual(
            "".join(text_deltas),
            "我来帮您查找酒店。感谢您的耐心等待！为您找到以下酒店。",
        )


if __name__ == "__main__":
    unittest.main()
