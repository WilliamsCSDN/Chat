from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_core.tools import tool

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skill"
_CONFIG_RE = re.compile(r"<!--\s*config\s*\n(.*?)\s*-->", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"(name|description|trigger|milvus_expr|keywords)\s*:\s*(.+)")


@dataclass
class SkillDef:
    name: str
    description: str
    trigger: str
    milvus_expr: str
    keywords: List[str]
    directory: str


def _parse_skills() -> List[SkillDef]:
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
        config_blocks = _CONFIG_RE.findall(content)
        for block in config_blocks:
            skill_name = ""
            description = ""
            trigger = ""
            milvus_expr = ""
            keywords: List[str] = []
            for line in block.strip().split("\n"):
                m = _KEY_VALUE_RE.match(line.strip())
                if not m:
                    continue
                key = m.group(1).strip()
                value = m.group(2).strip()
                if key == "name":
                    skill_name = value
                elif key == "description":
                    description = value
                elif key == "trigger":
                    trigger = value
                elif key == "milvus_expr":
                    milvus_expr = value
                elif key == "keywords":
                    keywords = [kw.strip() for kw in value.split(",") if kw.strip()]
            if skill_name and description and milvus_expr:
                skills.append(SkillDef(
                    name=skill_name,
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
