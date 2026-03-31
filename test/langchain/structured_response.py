from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from numba.scripts.generate_lower_listing import description
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from config.log import init_xxx
from config_settings import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME


init_xxx()

class Student(BaseModel):
    name: str = Field(description = "人名")
    email: str = Field(description = "个人邮箱")
    age: int = Field(description = "人的年龄")


agent = create_agent(
    model= ChatOpenAI(base_url=DASHSCOPE_BASE_URL, model= MODEL_NAME, api_key=DASHSCOPE_API_KEY, streaming=False),

    # system_prompt="you are a helpful assistant"
    response_format=ToolStrategy(Student)
)

if __name__ == '__main__':
    print("1")
    messages = [
        SystemMessage("you are a helpful assistant"),
        HumanMessage("我是一名学生，名字叫williams，邮箱：williamscsdn@gmail.com，年龄：28"),
    ]
    res = agent.invoke(
        {"messages":[{"role":"user", "content":"我是一名学生，名字叫williams，邮箱：williamscsdn@gmail.com，年龄：28"}]}
    )

    print(res)
    print(res.get("messages")[1].content)
    print(res["structured_response"])
