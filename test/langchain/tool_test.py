import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_openai import ChatOpenAI

from config.log import init_xxx
from langchain.agents import create_agent
from src.config.config_settings import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from src.tools.weather_tool import get_wether


init_xxx()

agent = create_agent(
    model= ChatOpenAI(base_url=DASHSCOPE_BASE_URL, model= MODEL_NAME, api_key=DASHSCOPE_API_KEY, streaming=False),
    tools=[get_wether],
    system_prompt="you are a helpful assistant",
)

if __name__ == '__main__':
    logging.info("登录成功")
    print("aasdf")

    agent.invoke({"messages":[{"role":"user","content":"深圳的天气如何"}]})


