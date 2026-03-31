from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from config_settings import DASHSCOPE_BASE_URL, MODEL_NAME, DASHSCOPE_API_KEY


@tool
def get_weather(city : str):
    """获取城市的天气状态"""
    return f"{city}天气不错的拉"

def init_agent(streaming: bool = True):

    agent = create_agent(
        model= ChatOpenAI(base_url=DASHSCOPE_BASE_URL, model= MODEL_NAME, api_key=DASHSCOPE_API_KEY, streaming=streaming),
        tools = [get_weather],
        system_prompt="you are a helpful assistant",
    )
    return agent