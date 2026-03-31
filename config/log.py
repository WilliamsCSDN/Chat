import logging
from gc import set_debug
from logging.handlers import RotatingFileHandler

def init_xxx():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[
            RotatingFileHandler("app.log", maxBytes=10 * 1024 * 1024, backupCount=5),  # 文件轮转
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    # 开启langchain的debug模式
    set_debug(True)

    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("openai").setLevel(logging.DEBUG)

