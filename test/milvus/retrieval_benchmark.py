"""
检索评测脚本：量化对比切分/重排策略
========================================
对比三种策略：
  A. baseline_char_window        (旧：字符滑窗切分 + 向量检索)
  B. sentence_window_only        (新：句级切分 + 向量检索)
  C. sentence_window_plus_rerank (新：句级切分 + 词面重排)

指标：
  - Hit@3 / Hit@5
  - MRR

使用：
  python3 retrieval_benchmark.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import milvus_test


@dataclass
class EvalQuery:
    query: str
    expected_chapter_num: int


EVAL_QUERIES: List[EvalQuery] = [
    EvalQuery("石猴出世花果山", 1),
    EvalQuery("拜菩提祖师学本领", 2),
    EvalQuery("东海龙宫借如意金箍棒", 3),
    EvalQuery("弼马温不入流愤而下界", 4),
    EvalQuery("偷吃蟠桃和仙丹", 5),
    EvalQuery("二郎神大战孙悟空", 5),
]


def baseline_chunk_xiyouji(raw: str, chunk_size: int = 150, overlap: int = 20) -> List[milvus_test.Chunk]:
    parts = milvus_test.CHAPTER_TITLE_SPLIT_PATTERN.split(raw.strip())
    all_chunks: List[milvus_test.Chunk] = []
    gid = 0

    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        content = parts[i + 1].strip()
        m = milvus_test.CHAPTER_NUM_PATTERN.search(title)
        chapter_num = milvus_test.cn2int(m.group(1)) if m else 0

        joined = "\n".join(p.strip() for p in content.split("\n") if p.strip())

        start, seq = 0, 0
        while start < len(joined):
            end = min(start + chunk_size, len(joined))
            text = joined[start:end].strip()
            if text:
                all_chunks.append(
                    milvus_test.Chunk(
                        chunk_id=gid,
                        chapter_num=chapter_num,
                        chapter_title=title,
                        seq=seq,
                        text=text,
                        char_count=len(text),
                    )
                )
                gid += 1
                seq += 1
            start += chunk_size - overlap

    return all_chunks


def local_vector_search(
    query: str,
    chunks: List[milvus_test.Chunk],
    embeddings: List[List[float]],
    top_k: int,
):
    model = milvus_test.load_embedding.load_embedding_model()
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    scored = sorted(
        ((milvus_test.cosine(qvec, embeddings[i]), chunks[i]) for i in range(len(chunks))),
        key=lambda x: -x[0],
    )
    return scored[:top_k]


def local_vector_search_with_rerank(
    query: str,
    chunks: List[milvus_test.Chunk],
    embeddings: List[List[float]],
    top_k: int,
):
    first_stage = local_vector_search(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        top_k=max(20, top_k * 6),
    )
    return milvus_test.rerank_candidates(query, first_stage, top_k=top_k)


def reciprocal_rank(hit_flags: List[bool]) -> float:
    for idx, ok in enumerate(hit_flags, start=1):
        if ok:
            return 1.0 / idx
    return 0.0


def evaluate_strategy(
    name: str,
    queries: List[EvalQuery],
    result_getter,
) -> Dict[str, float]:
    hit3 = 0
    hit5 = 0
    rr_sum = 0.0

    for q in queries:
        res3 = result_getter(q.query, 3)
        res5 = result_getter(q.query, 5)

        if res3 and isinstance(res3[0], tuple):
            flags3 = [item[1].chapter_num == q.expected_chapter_num for item in res3]
            flags5 = [item[1].chapter_num == q.expected_chapter_num for item in res5]
        else:
            flags3 = [item["chapter_num"] == q.expected_chapter_num for item in res3]
            flags5 = [item["chapter_num"] == q.expected_chapter_num for item in res5]

        hit3 += 1 if any(flags3) else 0
        hit5 += 1 if any(flags5) else 0
        rr_sum += reciprocal_rank(flags5)

    n = len(queries)
    return {
        "name": name,
        "hit@3": hit3 / n,
        "hit@5": hit5 / n,
        "mrr": rr_sum / n,
    }


def print_metrics(metrics: Dict[str, float]):
    print(
        f"{metrics['name']:<28}"
        f" Hit@3={metrics['hit@3']:.3f}"
        f"  Hit@5={metrics['hit@5']:.3f}"
        f"  MRR={metrics['mrr']:.3f}"
    )


def main():
    raw = milvus_test.load_xiyouji_text()

    print("=" * 72)
    print("Step 1: 构建两套切分")
    print("=" * 72)
    baseline_chunks = baseline_chunk_xiyouji(raw, chunk_size=150, overlap=20)
    improved_chunks = milvus_test.chunk_xiyouji(
        raw,
        target_chars=260,
        min_chars=140,
        overlap_sentences=2,
    )
    print(f"baseline chunks: {len(baseline_chunks)}")
    print(f"improved chunks: {len(improved_chunks)}")

    print("\n" + "=" * 72)
    print("Step 2: 向量化（可能较慢）")
    print("=" * 72)
    baseline_emb = milvus_test.embed_texts([c.text for c in baseline_chunks])
    improved_emb = milvus_test.embed_texts([c.text for c in improved_chunks])
    print(f"baseline embeddings: {len(baseline_emb)}")
    print(f"improved embeddings: {len(improved_emb)}")

    print("\n" + "=" * 72)
    print("Step 3: 检索评测（6 条标注查询）")
    print("=" * 72)
    m1 = evaluate_strategy(
        "baseline_char_window",
        EVAL_QUERIES,
        lambda query, k: local_vector_search(query, baseline_chunks, baseline_emb, k),
    )
    m2 = evaluate_strategy(
        "sentence_window_only",
        EVAL_QUERIES,
        lambda query, k: local_vector_search(query, improved_chunks, improved_emb, k),
    )
    m3 = evaluate_strategy(
        "sentence_window_plus_rerank",
        EVAL_QUERIES,
        lambda query, k: local_vector_search_with_rerank(query, improved_chunks, improved_emb, k),
    )

    print_metrics(m1)
    print_metrics(m2)
    print_metrics(m3)

    print("\n" + "=" * 72)
    print("Step 4: 单查询示例（方便你看直观命中）")
    print("=" * 72)
    sample = "东海龙宫借兵器"
    print(f"query: {sample}")
    baseline_top3 = local_vector_search(sample, baseline_chunks, baseline_emb, 3)
    rerank_top3 = local_vector_search_with_rerank(sample, improved_chunks, improved_emb, 3)

    print("\n[baseline top3]")
    for score, c in baseline_top3:
        print(f"- ch{c.chapter_num} score={score:.4f} {c.text[:40]}...")

    print("\n[improved + rerank top3]")
    for item in rerank_top3:
        print(
            f"- ch{item['chapter_num']} score={item['final_score']:.4f} "
            f"(vec={item['vector_score']:.4f}, lex={item['lexical_score']:.4f}) "
            f"{item['text'][:40]}..."
        )


if __name__ == "__main__":
    main()
