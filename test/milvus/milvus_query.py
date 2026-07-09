
from dataclasses import dataclass
import re

import load_embedding


# ─────────────────────────────────────────────────────────
# 1. 切分：章节 + 滑动窗口
# ─────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:      int
    chapter_num:   int
    chapter_title: str
    seq:           int       # 同章内的第几块
    text:          str
    char_count:    int

# ─────────────────────────────────────────────────────────
# 3. Milvus：建 Collection → 插入 → 建索引
# ─────────────────────────────────────────────────────────
COLLECTION = "xiyouji"
DIM        = 384

def init_milvus():
    from pymilvus import (
        connections, CollectionSchema, FieldSchema, DataType, Collection,
    )
    connections.connect(host="localhost", port="19530")
    print("✅ 已连接 Milvus")

    # if utility.has_collection(COLLECTION):
    #     Collection(COLLECTION).drop()
    #     print("🗑️  旧 Collection 已删除")

    fields = [
        FieldSchema("id",            DataType.INT64,          is_primary=True, auto_id=True),
        FieldSchema("chapter_num",   DataType.INT64),
        FieldSchema("chapter_title", DataType.VARCHAR,        max_length=200),
        FieldSchema("seq",           DataType.INT64),
        FieldSchema("text",          DataType.VARCHAR,        max_length=2000),
        FieldSchema("embedding",     DataType.FLOAT_VECTOR,   dim=DIM),
    ]
    col = Collection(COLLECTION, CollectionSchema(fields, description="西游记语义检索"))
    print(f"📦 Collection [{COLLECTION}] 创建成功")
    return col


# ─────────────────────────────────────────────────────────
# 4. 语义检索
# ─────────────────────────────────────────────────────────
def search_milvus(query: str, top_k: int = 3):
    from pymilvus import Collection

    vec = load_embedding.load_embedding_model().encode([query], normalize_embeddings=True).tolist()

    col = Collection(COLLECTION)
    col.load()
    recall_k = max(top_k * 6, 20)
    hits = col.search(
        data          = vec,
        anns_field    = "embedding",
        param         = {"metric_type": "COSINE", "params": {"nprobe": 32}},
        limit         = recall_k,
        output_fields = ["chapter_title", "chapter_num", "text"],
        expr = 'chapter_num == 2', # 精确查询
    )
    reranked = rerank_hits(query, hits[0], top_k=top_k)
    _print_results(query, reranked)


def build_query_terms(query: str):
    q = re.sub(r"\s+", "", query)
    if not q:
        return []
    terms = {q}
    for n in (2, 3):
        if len(q) >= n:
            for i in range(len(q) - n + 1):
                terms.add(q[i : i + n])
    return sorted(terms, key=len, reverse=True)


def lexical_overlap_score(text: str, terms):
    if not terms:
        return 0.0
    hit_terms = [t for t in terms if t in text]
    if not hit_terms:
        return 0.0
    coverage = len(hit_terms) / len(terms)
    weighted = sum(len(t) for t in hit_terms) / sum(len(t) for t in terms)
    return 0.6 * coverage + 0.4 * weighted


def rerank_hits(query: str, hits, top_k: int = 3):
    terms = build_query_terms(query)
    ranked = []
    for hit in hits:
        e = hit.entity
        lexical_score = lexical_overlap_score(e.text, terms)
        final_score = 0.75 * float(hit.score) + 0.25 * lexical_score
        ranked.append({
            "chapter_title": e.chapter_title,
            "text": e.text,
            "vector_score": float(hit.score),
            "lexical_score": lexical_score,
            "final_score": final_score,
        })
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_k]

def _print_results(query, hits):
    print(f"\n🔍 检索：「{query}」")
    print("─" * 56)
    for hit in hits:
        print(
            f"  [{hit['chapter_title']}]  综合分 {hit['final_score']:.4f}"
            f"（向量 {hit['vector_score']:.4f} / 词面 {hit['lexical_score']:.4f}）"
        )
        print(f"  {hit['text'][:80]}…\n")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
QUERIES = [
    # "孙悟空大闹天宫",
    # "拜师学艺学武功",
    # "借兵器去东海龙宫",
    # "唐僧一行人最终取得真经",
    # "孙悟空被压在五指山下",
    "孙悟空驾着狂风回到了花果山"
]

def main():
    init_milvus()

    # Step 4 检索演示
    print("=" * 56)
    print("  Step 4  语义检索演示")
    print("=" * 56)
    for q in QUERIES:
        search_milvus(q, top_k=3)


if __name__ == "__main__":
    main()