from openai import OpenAI
from openai.types.chat import ChatCompletion

from src.config.config_settings import DASHSCOPE_BASE_URL,DASHSCOPE_API_KEY,MODEL_NAME


def openaitest():
    client = OpenAI(base_url=DASHSCOPE_BASE_URL, api_key=DASHSCOPE_API_KEY)

    response: ChatCompletion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "你是谁"},
            {"role": "assistant", "content": "我是williams"},
            {"role": "user", "content": "你几岁"},
            {"role": "assistant", "content": "28"},
            {"role": "user", "content": "你是谁你几岁了"},
        ],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

if __name__ == '__main__':
    openaitest()
