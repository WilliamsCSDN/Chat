import logging
import os

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# 阿里云百炼 DashScope API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")

# 百炼 OpenAI 兼容模式的 base_url
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not DASHSCOPE_API_KEY:
    logger.warning(
        "DASHSCOPE_API_KEY is not set. "
        "Please set it in the environment or in a .env file. "
        "Chat API requests will fail until a valid key is provided."
    )

