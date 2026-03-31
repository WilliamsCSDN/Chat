import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_core.messages import AIMessageChunk, ToolMessage, ToolMessageChunk

from config._init_ import init_agent
from config.log import init_xxx

init_xxx()


if __name__ == '__main__':
    user_prompt = "深圳天气怎么样最近, 顺便写个一千字的ai新闻"
    print(f"User: {user_prompt}\n")

    agent = init_agent()
    seen_tool_names: set[str] = set()
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        stream_mode="messages",
    ):
        msg, _meta = chunk
        if isinstance(msg, AIMessageChunk):
            if msg.content:
                print(msg.content, end="", flush=True)
            for tc in msg.tool_calls or []:
                name = tc.get("name")
                if name and name not in seen_tool_names:
                    seen_tool_names.add(name)
                    print(f"\nCalling tools: [{name}]", flush=True)
        elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
            print(f"\n[Tool {msg.name} done]\n", end="", flush=True)

    print()
