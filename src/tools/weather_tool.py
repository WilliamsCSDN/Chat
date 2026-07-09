from langchain_core.tools import tool


@tool
def get_wether(city: str) -> str:
    """GET weather for a given city"""
    return f"{city}天气非常好"
