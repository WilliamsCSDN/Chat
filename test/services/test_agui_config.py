import unittest

from src.config.config_settings import AGENT_RECURSION_LIMIT
from src.services.agui import stream_agui_events


class _RecordingAgent:
    def __init__(self):
        self.config = None

    async def astream_events(self, _input, *, version, config):
        self.config = config
        if False:
            yield version


class AguiConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_uses_configured_recursion_limit(self):
        agent = _RecordingAgent()

        events = [
            event
            async for event in stream_agui_events(
                agent,
                [],
                "thread-1",
            )
        ]

        self.assertEqual(events, [])
        self.assertEqual(
            agent.config["recursion_limit"],
            AGENT_RECURSION_LIMIT,
        )


if __name__ == "__main__":
    unittest.main()
