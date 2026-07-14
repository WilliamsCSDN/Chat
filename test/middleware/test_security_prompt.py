import unittest

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.security_prompt import inject_security_prompt


class SecurityPromptTests(unittest.TestCase):
    def test_security_prompt_is_dynamic_and_does_not_mutate_state(self):
        human = HumanMessage(content="测试编排稳定性")
        state = {"messages": [human]}
        request = ModelRequest(model=object(), messages=[human], state=state)
        captured = []

        def handler(model_request):
            captured.append(model_request)
            return ModelResponse(result=[AIMessage(content="ok")])

        inject_security_prompt.wrap_model_call(request, handler)
        inject_security_prompt.wrap_model_call(request, handler)

        self.assertEqual(state["messages"], [human])
        self.assertTrue(
            all(item.system_message is not None for item in captured)
        )
        self.assertTrue(
            all(
                item.system_message.text.count("## 可用技能") == 1
                for item in captured
            )
        )


if __name__ == "__main__":
    unittest.main()
