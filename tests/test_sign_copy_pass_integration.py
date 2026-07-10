from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ExpandResultSignCopyTests(unittest.TestCase):
    def test_expand_result_runs_stage2_when_cues(self) -> None:
        from prompt_expander import PromptExpansionResult, expand_prompt_result

        stage1 = PromptExpansionResult(
            expanded="failing neon signs hanging low along abandoned alleyways in fog",
            changed=True,
            backend="comfy",
        )
        stage2_raw = (
            'UPDATED\n\nfailing neon signs reading "LAST CALL" hanging low along '
            "abandoned alleyways in fog"
        )

        with patch("prompt_expander.expand_prompt_local", return_value=stage1), patch(
            "comfy_qwen_vl.comfy_qwen_vl_available", return_value=True
        ), patch("comfy_qwen_vl.expand_prompt_comfy", return_value=stage2_raw) as expand:
            result = expand_prompt_result("short", backend="local")

        self.assertIn("LAST CALL", result.expanded)
        self.assertTrue(result.changed)
        self.assertTrue(result.sign_copy_pass and result.sign_copy_pass.get("changed"))
        self.assertFalse(expand.call_args.kwargs.get("free_vram", True))

    def test_expand_result_skips_stage2_glow_only(self) -> None:
        from prompt_expander import PromptExpansionResult, expand_prompt_result

        stage1 = PromptExpansionResult(
            expanded="a quiet street lit by neon glow and neon reflections on wet asphalt",
            changed=True,
            backend="comfy",
        )

        with patch("prompt_expander.expand_prompt_local", return_value=stage1), patch(
            "comfy_qwen_vl.expand_prompt_comfy"
        ) as expand:
            result = expand_prompt_result("short", backend="local")

        expand.assert_not_called()
        self.assertEqual(result.sign_copy_pass.get("skipped_reason"), "no_cues")
        self.assertNotIn("reading", result.expanded.lower())

    def test_expand_result_preserves_defined_paper_text(self) -> None:
        from prompt_expander import PromptExpansionResult, expand_prompt_result

        prompt = 'a woman holds a paper that reads "MEET ME AT DAWN" in a rainy alley'
        stage1 = PromptExpansionResult(expanded=prompt, changed=True, backend="openrouter")

        with patch("prompt_expander.expand_prompt_openrouter", return_value=stage1), patch(
            "comfy_qwen_vl.expand_prompt_comfy"
        ) as expand:
            result = expand_prompt_result("short", backend="openrouter")

        expand.assert_not_called()
        self.assertEqual(result.expanded, prompt)
        self.assertEqual(result.sign_copy_pass.get("skipped_reason"), "already_present")

    def test_stage1_error_skips_stage2(self) -> None:
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
        self.assertIsNone(result.sign_copy_pass)


if __name__ == "__main__":
    unittest.main()
