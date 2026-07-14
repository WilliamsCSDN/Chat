from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from typing import Tuple

from langchain_core.tools import tool

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skill"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(fm_text: str) -> dict[str, str]:
    """Parse simple YAML frontmatter for key: value pairs."""
    result: dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")
            result[key] = value
    return result


@dataclass
class SkillDef:
    """Anthropic-style skill definition with name + description from frontmatter.

    body holds the SKILL.md content without frontmatter,
    for progressive disclosure when a specific skill is loaded.
    """
    name: str
    description: str
    directory: str
    body: str


def _parse_skills() -> List[SkillDef]:
    """Parse all SKILL.md files, returning SkillDef list.

    Follows Anthropic's skill spec: reads name + description from YAML frontmatter.
    Body content (without frontmatter) is stored for progressive disclosure.

    Each skill directory must contain a SKILL.md with YAML frontmatter
    defining at minimum ``name`` and ``description``.
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

        fm_match = _FRONTMATTER_RE.match(content)
        if not fm_match:
            continue

        fm = _parse_frontmatter(fm_match.group(1))
        name = fm.get("name", skill_dir.name)
        description = fm.get("description", "")
        body = content[fm_match.end():].strip()

        if name:
            skills.append(SkillDef(
                name=name,
                description=description,
                directory=skill_dir.name,
                body=body,
            ))

    return skills


def load_skills_for_context() -> list[tuple[str, str]]:
    """Extract name and description for building system-prompt context."""
    return [(s.name, s.description) for s in _parse_skills()]


@tool
def skills_load(name: str = "") -> str:
    """List available skills. Pass a skill name to load its full instructions.

    Progressive disclosure: by default only names and descriptions are listed.
    When a specific skill is needed, pass its name to receive the full SKILL.md body.
    """
    skills = _parse_skills()
    if not skills:
        return "No skills available."

    if name:
        matched = [s for s in skills if s.name == name]
        if not matched:
            available = ", ".join(s.name for s in skills)
            return f"Skill '{name}' not found. Available: {available}"
        return matched[0].body

    # Progressive disclosure: list name + description only
    return "\n".join(
        f"- **{s.name}**: {s.description}" for s in skills
    )
