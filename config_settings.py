import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 阿里云百炼 DashScope API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-plus")

# 百炼 OpenAI 兼容模式的 base_url
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

