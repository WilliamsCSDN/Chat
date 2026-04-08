"""
西游记 × Milvus 完整案例
========================================
流程：
  1. 默认加载 xiyouji.txt（不存在时回退内置节选）
  2. 按章节 + 滑动窗口切分
  3. 用 sentence-transformers 生成中文向量
  4. 写入 Milvus，建索引
  5. 语义检索演示

依赖：
  pip install pymilvus sentence-transformers

Milvus 启动（docker）：
  docker run -d --name milvus-standalone \
    -p 19530:19530 -p 9091:9091 \
    milvusdb/milvus:v2.4.0 milvus run standalone
"""

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import load_embedding

# ─────────────────────────────────────────────────────────
# 0. 内置文本（xiyouji.txt 不存在时的回退示例）
# ─────────────────────────────────────────────────────────
RAW_TEXT = """
第一回 灵根育孕源流出 心性修持大道生

话说很古以前，天地未开，混沌初判，轻清者上升为天，重浊者下沉为地。
盘古开天辟地之后，三皇治世，五帝定伦，世界之间，遂分为四大部洲：
曰东胜神洲，曰西牛贺洲，曰南赡部洲，曰北俱芦洲。
东胜神洲有一国土，名曰傲来国。国近大海，海中有一座名山，唤为花果山。
那山正当顶上，有一块仙石，盖自开辟以来，每受天真地秀，日精月华，感之既久，
遂有灵通之意，内育仙胞，一日迸裂，产一石卵，似圆球样大，因见风化作一个石猴。
那猴在山中，却会行走跳跃，食草木，饮涧泉，采山花，觅树果；
与狼虫为伴，虎豹为群，獐鹿为友，猕猿为亲。
一朝天气炎热，与群猴避暑，都在松阴之下顽耍。
一群猴子耍了一会，却去那山涧中洗澡。见那股涧水奔流，
众猴拍手称扬道：这股水不知是哪里的水，我们今日赶闲无事，顺涧边往上溜头寻看源流。
他有本事的，钻进去寻个源头出来，不伤身体者，我等即拜他为王！
连呼了三声，忽见丛杂中跳出一个石猴，应声高叫道：我进去！我进去！
那石猴把眼睛闭了一下，纵身跳入瀑布泉中，忽然眼睁开，
抬头观看，见里面更无水波，宛然一座铁板桥，桥下之水冲贯于石窍之间。
石猴喜不自胜，一一看过，吃了些果食，拿起桥边一块大石，对准桥柱，
哗啦一声打了下去，那桥应声塌了，石猴从水中蹿出，高叫道：大造化！大造化！
众猴拱伏道：请大王表明仙居！石猴道：里面有花有树，有水有洞，可以安家。
众猴道：大王带我们进去！石猴当即将众猴引入水帘洞，称王为美猴王。

第二回 悟彻菩提真妙理 断魔归本合元神

话说美猴王自在山中，日月迅速，不觉七八年过去，忽然忧愁，
只怕有一日无常鬼至，去了这条性命。想起人有三教，道释与儒，
皆教人以仁义礼智，惟独道教，有长生不老之术。
美猴王遂弃了众猴，乘木筏漂洋过海，历尽艰辛，来到西牛贺洲。
寻得灵台方寸山，斜月三星洞，拜菩提老祖为师，赐名孙悟空。
老祖问：你学的是哪门课？悟空答：随祖师裁定。
祖师演示了三十六般变化，七十二地煞数，孙悟空苦学数年，皆已融会贯通。
一日，众师兄弟取笑，悟空被迫卖弄，惊动了老祖，唤至堂前，
老祖道：你这泼猴，在此学了多年，学了些什么邪道？你去吧！
孙悟空泪流满面，磕头辞别，驾起筋斗云飞回花果山。

第三回 四海千山皆拱伏 九幽十类尽除名

话说孙悟空回到花果山，见群猴受混世魔王欺压，大怒，
摆下演武场，聚合群猴操练武艺。因无称手兵器，前往东海龙宫求借。
东海龙王敖广先献大刀、方天戟，皆嫌太轻；
最后龙王献出海底镇海神针——定海神珍铁，重一万三千五百斤。
悟空用手拿来，叫声：再细些！那宝贝随口令缩细若绣花针，
收在耳朵里。复叫：粗长！又变作碗来粗细，二丈长短。
悟空持金箍棒，又向龙宫讨了披挂，身披黄金锁子甲，头顶凤翅紫金冠，
脚踏藕丝步云履，威风凛凛，驾云离去，回转花果山。
四海龙王告上天庭，玉帝欲讨伐，太白金星劝谏，愿领旨下界招安，化解此事。

第四回 官封弼马心何足 名注齐天意未宁

话说太白金星持节到花果山，宣玉帝旨意，请孙悟空上天任职。
悟空随天使腾云而至南天门，直入灵霄宝殿，拜见玉帝。
玉帝封悟空为弼马温，掌管御马监，养育天马。
悟空不知弼马温品级，在御马监尽心尽力，把那些天马喂养得膘肥体壮。
后问同僚方知弼马温不入流，大怒道：这般藐视老孙！推倒公案，
取出金箍棒，打出南天门，回花果山，竖起旗帜，自称齐天大圣。
玉帝大怒，命托塔天王李靖率十万天兵下界讨伐。
哪吒三太子与悟空交手，不分胜负；天兵大败，仍无人能制。
太白金星再次进言，招安封官，悟空遂被封为齐天大圣，管理蟠桃园。

第五回 乱蟠桃大圣偷丹 反天宫诸神捉怪

话说大圣在蟠桃园内，见那桃树上桃子又大又红，喜不自胜，
趁看园女童不注意，变作二寸高小人，藏于树梢，挑熟的吃了个饱。
王母娘娘设蟠桃盛会，请了各路神仙，却未请齐天大圣。
大圣怒道：我也是天宫一员，为何不请我？变作赤脚大仙模样，
混入瑶池，将玉液琼浆偷吃一空，又跑去太上老君炼丹炉边，
将葫芦内仙丹倒出，如吃炒豆一般，全数吃完。
酒醒后，知闯了大祸，急驾云离去。玉帝大怒，命十万天兵天将再次征讨，
摆下天罗地网。花果山一番激战，大圣七十二变神通，天兵奈何不得。
二郎神杨戬出马，与大圣斗法，旗鼓相当；太上老君从天上掷下金刚琢，
打中大圣天灵盖，方才拿住，押赴天宫候审。
"""


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

CN_NUM = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}
CHAPTER_TITLE_SPLIT_PATTERN = re.compile(r"(第[零〇一二三四五六七八九十百千\d]+回[^\n]*)\n")
CHAPTER_NUM_PATTERN = re.compile(r"第([零〇一二三四五六七八九十百千\d]+)回")

def cn2int(s: str) -> int:
    """章节号转整数：支持中文数字和阿拉伯数字"""
    s = s.strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)

    result, tmp = 0, 0
    for ch in s:
        if ch in CN_UNIT:
            v = CN_UNIT[ch]
            result += (tmp or 1) * v
            tmp = 0
        else:
            tmp = CN_NUM.get(ch, 0)
    return result + tmp

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？；!?])", text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    return sentences


def build_sentence_windows(
    sentences: List[str],
    target_chars: int = 260,
    min_chars: int = 140,
    overlap_sentences: int = 2,
) -> List[str]:
    windows: List[str] = []
    current: List[str] = []
    current_chars = 0

    def flush_window():
        nonlocal current, current_chars
        if not current:
            return
        merged = "".join(current).strip()
        if merged:
            windows.append(merged)
        overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
        current = overlap.copy()
        current_chars = sum(len(x) for x in current)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > target_chars * 2:
            if current:
                flush_window()
            for i in range(0, len(sent), target_chars):
                part = sent[i : i + target_chars].strip()
                if part:
                    windows.append(part)
            current = []
            current_chars = 0
            continue

        if current and current_chars + len(sent) > target_chars and current_chars >= min_chars:
            flush_window()

        current.append(sent)
        current_chars += len(sent)

    if current:
        merged = "".join(current).strip()
        if merged:
            windows.append(merged)

    return windows


def chunk_xiyouji(
    raw: str,
    target_chars: int = 260,
    min_chars: int = 140,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """
    ① 正则按"第X回"切成章节
    ② 章节内按句切分，再拼接成目标长度窗口
    ③ 使用句级重叠，减少语义被切断
    """
    parts = CHAPTER_TITLE_SPLIT_PATTERN.split(raw.strip())

    all_chunks: List[Chunk] = []
    gid = 0

    for i in range(1, len(parts) - 1, 2):
        title   = parts[i].strip()
        content = parts[i + 1].strip()

        m           = CHAPTER_NUM_PATTERN.search(title)
        chapter_num = cn2int(m.group(1)) if m else 0

        joined = ''.join(p.strip() for p in content.split('\n') if p.strip())
        sentences = split_sentences(joined)
        windows = build_sentence_windows(
            sentences,
            target_chars=target_chars,
            min_chars=min_chars,
            overlap_sentences=overlap_sentences,
        )

        for seq, text in enumerate(windows):
            all_chunks.append(Chunk(
                chunk_id      = gid,
                chapter_num   = chapter_num,
                chapter_title = title,
                seq           = seq,
                text          = text,
                char_count    = len(text),
            ))
            gid += 1

    return all_chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = load_embedding.load_embedding_model()
    vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vecs.tolist()


# ─────────────────────────────────────────────────────────
# 3. Milvus：建 Collection → 插入 → 建索引
# ─────────────────────────────────────────────────────────
COLLECTION = "xiyouji"
DIM        = 384

def init_milvus():
    from pymilvus import (
        connections, utility,
        CollectionSchema, FieldSchema, DataType, Collection,
    )
    connections.connect(host="localhost", port="19530")
    print("✅ 已连接 Milvus")

    if utility.has_collection(COLLECTION):
        Collection(COLLECTION).drop()
        print("🗑️  旧 Collection 已删除")

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

def insert_to_milvus(col, chunks: List[Chunk], embeddings: List[List[float]]):
    BATCH = 256
    for i in range(0, len(chunks), BATCH):
        b = chunks[i : i + BATCH]
        col.insert([
            [c.chapter_num   for c in b],
            [c.chapter_title for c in b],
            [c.seq           for c in b],
            [c.text          for c in b],
            embeddings[i : i + BATCH],
        ])
        print(f"  已插入 {min(i+BATCH, len(chunks))}/{len(chunks)} 条")
    col.flush()
    print(f"✅ 入库完成，总实体: {col.num_entities}")

def create_index(col):
    col.create_index(
        field_name   = "embedding",
        index_params = {
            "metric_type": "COSINE",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128},
        },
    )
    col.load()
    print("✅ 索引已建立，Collection 已加载到内存")


# ─────────────────────────────────────────────────────────
# 4. 语义检索
# ─────────────────────────────────────────────────────────
def build_query_terms(query: str) -> List[str]:
    q = re.sub(r"\s+", "", query)
    if not q:
        return []

    terms = {q}
    for n in (2, 3):
        if len(q) >= n:
            for i in range(len(q) - n + 1):
                terms.add(q[i : i + n])
    return sorted(terms, key=len, reverse=True)


def lexical_overlap_score(text: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    hit_terms = [t for t in terms if t in text]
    if not hit_terms:
        return 0.0
    coverage = len(hit_terms) / len(terms)
    weighted = sum(len(t) for t in hit_terms) / sum(len(t) for t in terms)
    return 0.6 * coverage + 0.4 * weighted


def rerank_candidates(query: str, candidates: List[Tuple[float, Chunk]], top_k: int) -> List[Dict]:
    terms = build_query_terms(query)
    reranked = []
    for vector_score, chunk in candidates:
        lexical_score = lexical_overlap_score(chunk.text, terms)
        final_score = 0.75 * vector_score + 0.25 * lexical_score
        reranked.append({
            "chapter_title": chunk.chapter_title,
            "chapter_num": chunk.chapter_num,
            "text": chunk.text,
            "vector_score": vector_score,
            "lexical_score": lexical_score,
            "final_score": final_score,
        })
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked[:top_k]


def search_milvus(query: str, top_k: int = 3):
    from pymilvus import Collection

    model = load_embedding.load_embedding_model()
    vec   = model.encode([query], normalize_embeddings=True).tolist()

    col = Collection(COLLECTION)
    col.load()
    recall_k = max(top_k * 6, 20)
    hits = col.search(
        data          = vec,
        anns_field    = "embedding",
        param         = {"metric_type": "COSINE", "params": {"nprobe": 32}},
        limit         = recall_k,
        output_fields = ["chapter_title", "chapter_num", "text"],
    )
    candidates = []
    for hit in hits[0]:
        e = hit.entity
        candidates.append((
            float(hit.score),
            Chunk(
                chunk_id=-1,
                chapter_num=e.chapter_num,
                chapter_title=e.chapter_title,
                seq=-1,
                text=e.text,
                char_count=len(e.text),
            ),
        ))
    reranked = rerank_candidates(query, candidates, top_k=top_k)
    _print_results(query, reranked)

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
# 5. 无 Milvus 时的纯 Python 余弦检索（备用演示）
# ─────────────────────────────────────────────────────────
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-9)

def search_local(query: str, chunks: List[Chunk],
                 embeddings: List[List[float]], top_k: int = 3):
    model = load_embedding.load_embedding_model()
    vec   = model.encode([query], normalize_embeddings=True)[0].tolist()

    scored = sorted(
        [(cosine(vec, embeddings[i]), chunks[i]) for i in range(len(chunks))],
        key=lambda x: -x[0],
    )
    reranked = rerank_candidates(query, scored[: max(top_k * 6, 20)], top_k=top_k)
    print(f"\n🔍 检索（本地）：「{query}」")
    print("─" * 56)
    for hit in reranked:
        print(
            f"  [{hit['chapter_title']}]  综合分 {hit['final_score']:.4f}"
            f"（向量 {hit['vector_score']:.4f} / 词面 {hit['lexical_score']:.4f}）"
        )
        print(f"  {hit['text'][:80]}…\n")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
QUERIES = [
    "孙悟空大闹天宫",
    "拜师学艺学武功",
    "借兵器去东海龙宫",
]

SOURCE_TXT = Path(__file__).resolve().parent / "xiyouji.txt"


def load_xiyouji_text() -> str:
    if SOURCE_TXT.exists():
        print(f"📄 加载文本文件: {SOURCE_TXT}")
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                return SOURCE_TXT.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "xiyouji.txt",
            b"",
            0,
            1,
            "无法使用 utf-8/utf-8-sig/gb18030/gbk 解码，请确认文件编码",
        )

    print(f"⚠️  未找到 {SOURCE_TXT.name}，回退使用内置示例文本")
    return RAW_TEXT


def main():
    raw_text = load_xiyouji_text()

    # Step 1 切分
    print("=" * 56)
    print("  Step 1  切分西游记")
    print("=" * 56)
    chunks = chunk_xiyouji(raw_text, target_chars=260, min_chars=140, overlap_sentences=2)
    avg_len = sum(c.char_count for c in chunks) / len(chunks)
    print(f"  总 chunk 数  : {len(chunks)}")
    print(f"  章节数       : {len(set(c.chapter_num for c in chunks))}")
    print(f"  平均长度     : {avg_len:.1f} 字\n")
    print("── 前 3 个 chunk 预览 ──")
    for c in chunks[:3]:
        print(f"  [{c.chapter_title}] seq={c.seq}  {c.char_count}字")
        print(f"  {c.text[:60]}…\n")

    # Step 2 向量化
    print("=" * 56)
    print("  Step 2  生成向量（sentence-transformers）")
    print("=" * 56)
    embeddings = embed_texts([c.text for c in chunks])
    print(f"✅ 向量维度: {len(embeddings[0])}，共 {len(embeddings)} 条\n")

    # Step 3 Milvus 入库
    print("=" * 56)
    print("  Step 3  写入 Milvus")
    print("=" * 56)
    use_milvus = False
    try:
        col = init_milvus()
        insert_to_milvus(col, chunks, embeddings)
        create_index(col)
        use_milvus = True
    except Exception as e:
        print(f"⚠️  Milvus 不可用（{e}）")
        print("   → 改用本地余弦检索演示\n")

    # Step 4 检索演示
    print("=" * 56)
    print("  Step 4  语义检索演示")
    print("=" * 56)
    for q in QUERIES:
        if use_milvus:
            search_milvus(q, top_k=3)
        else:
            search_local(q, chunks, embeddings, top_k=3)


if __name__ == "__main__":
    main()