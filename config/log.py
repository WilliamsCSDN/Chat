import logging
from logging.handlers import RotatingFileHandler


def init_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    log_file_path: str = "app.log",
    sdk_http_debug: bool = True,
) -> None:
    root_logger = logging.getLogger()
    log_level = getattr(logging, level, logging.INFO)
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        handlers = [logging.StreamHandler()]
        if log_to_file:
            handlers.append(
                RotatingFileHandler(
                    log_file_path,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=handlers,
        )

    if sdk_http_debug:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)


def init_xxx() -> None:
    """兼容旧调用入口，内部转发到新日志初始化函数。"""
    init_logging()

