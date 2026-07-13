from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_core.tools import tool

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skill"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_CATEGORY_RE = re.compile(r"###\s+(.+?)\s*\n(.*?)(?=\n###|\n##|\Z)", re.DOTALL)
_DESC_RE = re.compile(r"\*\*描述\*\*[：:]\s*(.+)")
_KEYWORDS_RE = re.compile(r"\*\*触发关键词\*\*[：:]\s*(.+)")
_MILVUS_RE = re.compile(r"\*\*Milvus\s*过滤表达式\*\*[：:]\s*(.+)")


@dataclass
class SkillDef:
    name: str
    description: str
    trigger: str
    milvus_expr: str
    keywords: List[str]
    directory: str


def _parse_skills() -> List[SkillDef]:
    """解析所有 skill 目录中的 SKILL.md，返回 SkillDef 列表。

    每个 SKILL.md 使用 YAML frontmatter 定义元数据，body 中的 ``## 分类定义``
    章节列出各知识分类。每个分类生成一个 SkillDef。
    """
    if not _SKILLS_DIR.is_dir():
        return []

    skills: List[SkillDef] = []
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue

        content = skill_md.read_text(encoding="utf-8")

        # 提取 YAML frontmatter
        fm_match = _FRONTMATTER_RE.match(content)
        if not fm_match:
            continue

        # 从 frontmatter 之后的正文中提取分类定义
        body = content[fm_match.end():]

        # 定位到 ## 分类定义 之后的内容
        category_start = body.find("## 分类定义")
        if category_start == -1:
            continue
        category_body = body[category_start:]

        for m in _CATEGORY_RE.finditer(category_body):
            category_name = m.group(1).strip()
            block = m.group(2)

            desc_match = _DESC_RE.search(block)
            kw_match = _KEYWORDS_RE.search(block)
            mv_match = _MILVUS_RE.search(block)

            if not (desc_match and mv_match):
                continue

            description = desc_match.group(1).strip()
            milvus_expr = mv_match.group(1).strip().strip("`")
            keywords: List[str] = []
            if kw_match:
                keywords = [kw.strip() for kw in re.split(r"[、,]", kw_match.group(1)) if kw.strip()]

            trigger = f'在「{category_name}」分类下检索相关知识'

            if category_name and description and milvus_expr:
                skills.append(SkillDef(
                    name=category_name,
                    description=description,
                    trigger=trigger,
                    milvus_expr=milvus_expr,
                    keywords=keywords,
                    directory=skill_dir.name,
                ))

    return skills


def load_skills_for_context() -> list[tuple[str, str]]:
    """提取 name、description 和 trigger，用于构建上下文。"""
    return [(s.name, s.description) for s in _parse_skills()]


@tool
def skills_load(name: str = "") -> str:
    """列出当前可用的知识库检索技能。传入 name 可加载指定技能。

    根据返回的技能列表，选择最匹配用户问题的技能
    """
    skills = _parse_skills()
    if not skills:
        return "当前没有可用的知识库技能。"
    if name:
        matched = [s for s in skills if s.name == name]
        if not matched:
            available = "、".join(s.name for s in skills)
            return f"未找到名为 `{name}` 的技能。当前可用技能：{available}"
        skill = matched[0]
        lines: list[str] = [f"- **{skill.name}**: {skill.description}\n  → {skill.trigger}"]
        return "\n".join(lines)

    lines: list[str] = []
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description} | {skill.trigger}")
    return "\n".join(lines)
