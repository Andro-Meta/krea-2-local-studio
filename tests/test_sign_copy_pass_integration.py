from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ExpandResultSinglePassTests(unittest.TestCase):
    def test_expand_result_returns_stage1_without_second_llm_call(self) -> None:
        from prompt_expander import PromptExpansionResult, expand_prompt_result

        stage1 = PromptExpansionResult(
            expanded=(
                'an anthropomorphic pickle holding a whiskey glass, wearing a vintage shirt '
                'with bold distressed lettering reading "BACK"'
            ),
            changed=True,
            backend="comfy",
        )

        with patch("prompt_expander.expand_prompt_local", return_value=stage1), patch(
            "comfy_qwen_vl.expand_prompt_comfy"
        ) as expand:
            result = expand_prompt_result(
                'a pickle holding a whiskey glass and wearing a shirt reading "BACK"',
                backend="local",
            )

        self.assertEqual(result.expanded, stage1.expanded)
        self.assertTrue(result.changed)
        self.assertIsNone(result.sign_copy_pass)
        expand.assert_not_called()

    def test_expand_result_does_not_retry_after_stage1_error(self) -> None:
        from prompt_expander import PromptExpansionResult, expand_prompt_result

        stage1 = PromptExpansionResult(
            expanded="original",
            changed=False,
            error="Comfy down",
            backend="comfy",
        )

        with patch("prompt_expander.expand_prompt_local", return_value=stage1), patch(
            "comfy_qwen_vl.expand_prompt_comfy"
        ) as expand:
            result = expand_prompt_result("original", backend="local")

        expand.assert_not_called()
        self.assertEqual(result.error, "Comfy down")
        self.assertIsNone(result.sign_copy_pass)


if __name__ == "__main__":
    unittest.main()
