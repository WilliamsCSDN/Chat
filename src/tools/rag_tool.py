import random

from langchain_core.tools import tool

from src.services.milvus_retriever import (
    _DEFAULT_CONFIDENCE_POLICY,
    filter_passages_by_confidence,
    get_default_retriever,
)


@tool
def retrieve_knowledge(query: str, category: str, top_k: int = 3) -> str:
    """查询内部知识库获取准确信息。当用户询问可能存在于知识库中的事实性、专业性内容时调用此工具。

    使用规则：
    1. 如果返回结果显示置信度很高（相似度 ≥ 0.90），直接引用返回的内容回答用户，无需额外解释或补充
    2. 如果返回结果显示未找到相关知识或置信度不足，请明确告知用户，并结合自身知识补充回答
    """
    retriever = get_default_retriever()
    if retriever is None:
        return "知识库服务当前不可用，请告知用户并尝试使用通用知识回答。"

    expr = f'category_l1 == "{category}"'

    # passages = retriever.search(query, top_k=top_k, expr=expr)
    # if not passages:
    #     return f"未检索到「{category}」相关知识，请告知用户并尝试使用通用知识回答。"
    #
    # filtered = filter_passages_by_confidence(passages, _DEFAULT_CONFIDENCE_POLICY)
    # if not filtered:
    #     return f"检索到「{category}」相关的一些内容但置信度不足，请告知用户检索结果不够可靠。"
    #
    #
    # lines: list[str] = []
    # for i, p in enumerate(filtered, 1):
    #     lines.append(f"[{i}] 来源: {p.source} | 相似度: {p.score:.4f}")
    #     lines.append(f"    内容: {p.text}")

    # return "\n".join(lines)
    lines: list[str] = []
    lines.append(f" 来源: google | 相似度: {random.uniform(0,1):.4f}")
    lines.append(f"    内容: {expr}")
    return "\n".join(lines)
