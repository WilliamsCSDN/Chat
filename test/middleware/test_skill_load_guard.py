import unittest

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.skill_load_guard import (
    prevent_duplicate_skill_load,
    skill_loaded_in_current_turn,
)


def _skill_call(skill_name: str, call_id: str) -> dict:
    return {
        "name": "skills_load",
        "args": {"name": skill_name},
        "id": call_id,
        "type": "tool_call",
    }


class SkillLoadGuardTests(unittest.TestCase):
    def test_detects_same_skill_loaded_after_latest_human_message(self):
        messages = [
            HumanMessage(content="测试"),
            AIMessage(
                content="",
                tool_calls=[_skill_call("orchestration-stress", "old-call")],
            ),
        ]

        self.assertTrue(
            skill_loaded_in_current_turn(
                messages,
                "orchestration-stress",
                current_tool_call_id="new-call",
            )
        )

    def test_allows_different_skill_in_same_turn(self):
        messages = [
            HumanMessage(content="测试"),
            AIMessage(
                content="",
                tool_calls=[_skill_call("orchestration-stress", "old-call")],
            ),
        ]

        self.assertFalse(
            skill_loaded_in_current_turn(
                messages,
                "rag-categories",
                current_tool_call_id="new-call",
            )
        )

    def test_allows_same_skill_after_new_human_message(self):
        messages = [
            HumanMessage(content="测试"),
            AIMessage(
                content="",
                tool_calls=[_skill_call("orchestration-stress", "old-call")],
            ),
            HumanMessage(content="继续编排测试"),
        ]

        self.assertFalse(
            skill_loaded_in_current_turn(
                messages,
                "orchestration-stress",
                current_tool_call_id="new-call",
            )
        )

    def test_duplicate_call_is_short_circuited(self):
        previous_call = _skill_call("orchestration-stress", "old-call")
        current_call = _skill_call("orchestration-stress", "new-call")
        state = {
            "messages": [
                HumanMessage(content="测试"),
                AIMessage(content="", tool_calls=[previous_call]),
                AIMessage(content="", tool_calls=[current_call]),
            ]
        }
        request = ToolCallRequest(
            tool_call=current_call,
            tool=None,
            state=state,
            runtime=None,
        )
        handler_called = False

        def handler(_request):
            nonlocal handler_called
            handler_called = True
            return ToolMessage(content="loaded", tool_call_id="new-call")

        result = prevent_duplicate_skill_load.wrap_tool_call(request, handler)

        self.assertFalse(handler_called)
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("已在当前用户轮次加载", result.content)


class AsyncSkillLoadGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_call_is_short_circuited_for_async_agent(self):
        previous_call = _skill_call("orchestration-stress", "old-call")
        current_call = _skill_call("orchestration-stress", "new-call")
        state = {
            "messages": [
                HumanMessage(content="测试"),
                AIMessage(content="", tool_calls=[previous_call]),
                AIMessage(content="", tool_calls=[current_call]),
            ]
        }
        request = ToolCallRequest(
            tool_call=current_call,
            tool=None,
            state=state,
            runtime=None,
        )
        handler_called = False

        async def handler(_request):
            nonlocal handler_called
            handler_called = True
            return ToolMessage(content="loaded", tool_call_id="new-call")

        result = await prevent_duplicate_skill_load.awrap_tool_call(
            request,
            handler,
        )

        self.assertFalse(handler_called)
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("已在当前用户轮次加载", result.content)


if __name__ == "__main__":
    unittest.main()
