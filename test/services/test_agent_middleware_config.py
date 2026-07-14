import unittest

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain_openai import ChatOpenAI

from src.config.config_settings import (
    AGENT_MODEL_CALL_LIMIT,
    SUMMARY_KEEP_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    SUMMARY_TRIGGER_TOKENS,
)
from src.middleware.security_prompt import inject_security_prompt
from src.middleware.skill_load_guard import prevent_duplicate_skill_load
from src.services.chat_service import _build_agent_middleware


class AgentMiddlewareConfigTests(unittest.TestCase):
    def test_agent_uses_bounded_non_mutating_middleware(self):
        chat_model = ChatOpenAI(model="test-model", api_key="test-key")

        middleware = _build_agent_middleware(chat_model)

        self.assertIn(inject_security_prompt, middleware)
        self.assertIn(prevent_duplicate_skill_load, middleware)

        model_limit = next(
            item
            for item in middleware
            if isinstance(item, ModelCallLimitMiddleware)
        )
        self.assertEqual(model_limit.run_limit, AGENT_MODEL_CALL_LIMIT)
        self.assertEqual(model_limit.exit_behavior, "end")

        summary = next(
            item
            for item in middleware
            if isinstance(item, SummarizationMiddleware)
        )
        self.assertEqual(
            summary.trigger,
            [
                ("tokens", SUMMARY_TRIGGER_TOKENS),
                ("messages", SUMMARY_TRIGGER_MESSAGES),
            ],
        )
        self.assertEqual(
            summary.keep,
            ("messages", SUMMARY_KEEP_MESSAGES),
        )


if __name__ == "__main__":
    unittest.main()
