import unittest
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "skill"
    / "orchestration-stress"
    / "SKILL.md"
)


class OrchestrationStressSkillTests(unittest.TestCase):
    def test_skill_has_turn_scoped_branches_and_stop_contract(self):
        content = SKILL_PATH.read_text(encoding="utf-8")

        self.assertIn("当前用户轮次只能加载一次", content)
        self.assertIn("首次运行分支", content)
        self.assertIn("继续编排测试分支", content)
        self.assertIn("完成阶段 4 后不得再调用任何工具", content)
        self.assertIn("输出最终回复并立即停止", content)


if __name__ == "__main__":
    unittest.main()
