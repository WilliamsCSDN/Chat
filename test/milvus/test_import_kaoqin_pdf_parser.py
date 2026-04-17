from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MILVUS_TEST_DIR = PROJECT_ROOT / "test" / "milvus"
if str(MILVUS_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(MILVUS_TEST_DIR))

import import_kaoqin_pdf as kaoqin_import


class KaoqinPdfParserTests(unittest.TestCase):
    def test_is_heading_line_should_not_treat_numbered_items_as_heading(self):
        self.assertFalse(kaoqin_import.is_heading_line("1. 公平公正原则"))
        self.assertFalse(kaoqin_import.is_heading_line("2. 严格管理原则"))
        self.assertFalse(kaoqin_import.is_heading_line("3. 实事求是原则"))

    def test_extract_structured_sections_keeps_numbered_items_in_section_content(self):
        page_texts = [
            "\n".join(
                [
                    "第一章 总则",
                    "第一条 目的",
                    "为规范公司员工考勤管理。",
                    "第三条 基本原则",
                    "1. 公平公正原则",
                    "2. 严格管理原则",
                    "第二章 工作时间",
                    "第四条 标准工作时间",
                    "1. 公司实行标准工时制。",
                ]
            )
        ]
        sections = kaoqin_import.extract_structured_sections(page_texts)
        section_map = {section.title: section.content for section in sections}

        self.assertIn("第三条 基本原则", section_map)
        self.assertIn("1. 公平公正原则", section_map["第三条 基本原则"])
        self.assertIn("2. 严格管理原则", section_map["第三条 基本原则"])
        self.assertIn("第四条 标准工作时间", section_map)
        self.assertIn("1. 公司实行标准工时制。", section_map["第四条 标准工作时间"])


if __name__ == "__main__":
    unittest.main()
