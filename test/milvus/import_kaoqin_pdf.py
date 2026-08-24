from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

import load_embedding

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - 由 collect_pdf_pages 在运行时抛出更明确错误
    PdfReader = None


DEFAULT_PDF_PATH = Path(__file__).resolve().parent / "kaoqin.pdf"
DEFAULT_COLLECTION = "kaoqin_pdf"
DEFAULT_DIM = 1024
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_SENTENCES = 6


@dataclass
class PdfChunkRecord:
    doc_name: str
    page_no: int
    chunk_no: int
    section_title: str
    text: str
    searchable_text: str


@dataclass
class SectionBlock:
    title: str
    page_no: int
    content: str


def normalize_text(value: str, max_len: int = 8192) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_len]


HEADING_PATTERN = re.compile(r"^(第[零〇一二三四五六七八九十百千\d]+[章节条款])(?:\s+(.+))?$")


def parse_heading_line(line: str) -> tuple[str, str] | None:
    text = normalize_text(line, max_len=500)
    if not text:
        return None

    matched = HEADING_PATTERN.match(text)
    if not matched:
        return None

    heading_token = matched.group(1).strip()
    rest = normalize_text(matched.group(2) or "", max_len=500)
    if not rest:
        return heading_token, ""

    title_suffix = rest
    inline_content = ""

    split_by_colon = re.match(r"^(.{1,40}?)[：:](.+)$", rest)
    if split_by_colon:
        title_suffix = normalize_text(split_by_colon.group(1), max_len=120)
        inline_content = normalize_text(split_by_colon.group(2), max_len=500)
    else:
        list_marker = re.search(r"(?:^|\s)(?:\d+\.|[（(][一二三四五六七八九十\d]+[)）])\s*", rest)
        if list_marker:
            marker_start = list_marker.start()
            marker_text = list_marker.group(0).strip()
            before_marker = normalize_text(rest[:marker_start], max_len=120)
            after_marker = normalize_text(rest[marker_start + len(marker_text) :], max_len=500)
            if before_marker:
                title_suffix = before_marker
            else:
                title_suffix = ""
            inline_content = normalize_text(f"{marker_text} {after_marker}", max_len=500)
        elif len(rest) > 24 and " " in rest:
            maybe_title, maybe_content = rest.split(" ", 1)
            if len(maybe_title) <= 12 and maybe_content:
                title_suffix = normalize_text(maybe_title, max_len=120)
                inline_content = normalize_text(maybe_content, max_len=500)

    heading_title = heading_token if not title_suffix else f"{heading_token} {title_suffix}"
    return heading_title.strip(), inline_content.strip()


def is_heading_line(line: str) -> bool:
    return parse_heading_line(line) is not None


def split_to_sentences(text: str) -> List[str]:
    units: List[str] = []
    for raw_line in text.split("\n"):
        line = normalize_text(raw_line, max_len=20000)
        if not line:
            continue
        parts = re.split(r"(?<=[。！？；!?;])", line)
        for part in parts:
            sentence = part.strip()
            if sentence:
                units.append(sentence)
    return units


def split_section_to_chunks(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> List[str]:
    if max_sentences <= 0:
        raise ValueError("max_sentences 必须 > 0")
    sentences = split_to_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= max_sentences:
            chunks.append(normalize_text("".join(current), max_len=8192))
            current = []
    if current:
        chunks.append(normalize_text("".join(current), max_len=8192))
    return [chunk for chunk in chunks if chunk]


def extract_structured_sections(page_texts: List[str]) -> List[SectionBlock]:
    sections: List[SectionBlock] = []
    current_title = "导言"
    current_page_no = 1
    current_lines: List[str] = []

    def flush() -> None:
        if not current_lines:
            return
        content = normalize_text("\n".join(current_lines), max_len=200000)
        if content:
            sections.append(
                SectionBlock(
                    title=current_title,
                    page_no=current_page_no,
                    content=content,
                )
            )

    for page_index, page_text in enumerate(page_texts, start=1):
        normalized_page = normalize_text(page_text, max_len=200000)
        if not normalized_page:
            continue
        for raw_line in normalized_page.split("\n"):
            line = normalize_text(raw_line, max_len=20000)
            if not line:
                continue
            heading = parse_heading_line(line)
            if heading:
                flush()
                heading_title, inline_content = heading
                current_title = heading_title
                current_page_no = page_index
                current_lines = []
                if inline_content:
                    current_lines.append(inline_content)
                continue
            current_lines.append(line)

    flush()
    return sections


def collect_pdf_pages(pdf_path: Path) -> List[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    if PdfReader is None:
        raise ImportError("缺少 pypdf 依赖，请执行 `uv add pypdf` 后重试")

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        page_text = normalize_text(page.extract_text() or "", max_len=200000)
        if page_text:
            pages.append(page_text)
    return pages


def build_chunk_records(
    doc_name: str,
    page_texts: List[str],
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> List[PdfChunkRecord]:
    records: List[PdfChunkRecord] = []
    sections = extract_structured_sections(page_texts)
    for section in sections:
        chunks = split_section_to_chunks(section.content, max_sentences=max_sentences)
        for chunk_index, chunk in enumerate(chunks, start=1):
            searchable_text = normalize_text(
                f"文档:{doc_name}\n页码:{section.page_no}\n章节:{section.title}\n内容:{chunk}",
                max_len=8192,
            )
            records.append(
                PdfChunkRecord(
                    doc_name=doc_name,
                    page_no=section.page_no,
                    chunk_no=chunk_index,
                    section_title=section.title,
                    text=normalize_text(chunk, max_len=8192),
                    searchable_text=searchable_text,
                )
            )
    return records


def init_collection(collection_name: str, dim: int, drop_existing: bool) -> Collection:
    if utility.has_collection(collection_name):
        if drop_existing:
            Collection(collection_name).drop()
        else:
            collection = Collection(collection_name)
            field_names = {field.name for field in collection.schema.fields}
            required_fields = {"doc_name", "page_no", "chunk_no", "section_title", "text", "searchable_text", "embedding"}
            if not required_fields.issubset(field_names):
                raise ValueError(
                    f"Collection `{collection_name}` 的 schema 与当前脚本不兼容，请加 --drop-existing 重建"
                )
            return collection

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("doc_name", DataType.VARCHAR, max_length=256),
        FieldSchema("page_no", DataType.INT64),
        FieldSchema("chunk_no", DataType.INT64),
        FieldSchema("section_title", DataType.VARCHAR, max_length=512),
        FieldSchema("text", DataType.VARCHAR, max_length=8192),
        FieldSchema("searchable_text", DataType.VARCHAR, max_length=8192),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Kaoqin PDF chunks")
    return Collection(collection_name, schema=schema)


def insert_records(collection: Collection, records: List[PdfChunkRecord], batch_size: int) -> None:
    if not records:
        raise ValueError("没有可入库的 PDF 分片数据")

    model = load_embedding.load_embedding_model()
    encoded = model.encode(
        [record.searchable_text for record in records],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        collection.insert(
            [
                [r.doc_name for r in batch],
                [r.page_no for r in batch],
                [r.chunk_no for r in batch],
                [r.section_title for r in batch],
                [r.text for r in batch],
                [r.searchable_text for r in batch],
                batch_embeddings,
            ]
        )
        print(f"已插入: {min(i + batch_size, len(records))}/{len(records)}")
    collection.flush()


def ensure_index(collection: Collection) -> None:
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.load()


def main():
    parser = argparse.ArgumentParser(description="切分 kaoqin.pdf 并导入 Milvus")
    parser.add_argument("--pdf-path", type=Path, default=DEFAULT_PDF_PATH, help="PDF 文件路径")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection 名")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--drop-existing", action="store_true", help="导入前删除同名 collection")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批量写入大小")
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES, help="每个分片最多句子数")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {args.pdf_path}")
    if args.max_sentences <= 0:
        raise ValueError("max-sentences 必须 > 0")

    print(f"读取 PDF: {args.pdf_path}")
    page_texts = collect_pdf_pages(args.pdf_path)
    if not page_texts:
        raise ValueError("PDF 无可提取文本，请确认是可复制文本而非纯扫描图片")

    records = build_chunk_records(
        doc_name=args.pdf_path.name,
        page_texts=page_texts,
        max_sentences=args.max_sentences,
    )
    if not records:
        raise ValueError("PDF 切分后无有效分片")
    print(f"总页数: {len(page_texts)}，总分片数: {len(records)}")

    print(f"连接 Milvus: {args.host}:{args.port}")
    connections.connect(alias="default", host=args.host, port=args.port)
    collection = init_collection(args.collection, dim=DEFAULT_DIM, drop_existing=args.drop_existing)
    print(f"Collection 就绪: {args.collection}")

    insert_records(collection, records, batch_size=args.batch_size)
    ensure_index(collection)
    print(f"导入完成，当前实体数: {collection.num_entities}")


if __name__ == "__main__":
    main()
