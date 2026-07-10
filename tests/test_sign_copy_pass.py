from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sign_copy_pass import (  # noqa: E402
    accept_sign_copy_output,
    needs_sign_copy,
    parse_sign_copy_protocol,
    run_sign_copy_pass,
    sign_copy_already_present,
)


class SignCopyCueTests(unittest.TestCase):
    def test_glow_only_does_not_need_copy(self) -> None:
        self.assertFalse(needs_sign_copy("neon glow and neon reflections on wet asphalt"))

    def test_neon_sign_needs_copy(self) -> None:
        self.assertTrue(needs_sign_copy("failing neon signs hanging low along abandoned alleyways"))

    def test_held_paper_without_words_needs_copy(self) -> None:
        self.assertTrue(needs_sign_copy("a woman holds a handwritten paper toward the camera"))

    def test_paper_with_defined_words_already_satisfied(self) -> None:
        prompt = 'a woman holds a paper that reads "MEET ME AT DAWN"'
        self.assertTrue(sign_copy_already_present(prompt))

    def test_blank_or_unreadable_paper_skips(self) -> None:
        self.assertFalse(needs_sign_copy("a folded blank letter face-down on the table"))

    def test_sign_with_quotes_already_satisfied(self) -> None:
        self.assertTrue(sign_copy_already_present('a red neon sign reading "OPEN"'))

    def test_clutter_papers_skip(self) -> None:
        self.assertFalse(needs_sign_copy("debris-strewn corners with discarded cardboard boxes and stacks of papers"))

    def test_taco_style_unsigned_neon(self) -> None:
        prompt = (
            "a walking taco under bruised twilight, pale orange glow of failing neon signs "
            "hanging low along abandoned alleyways"
        )
        self.assertTrue(needs_sign_copy(prompt))
        self.assertFalse(sign_copy_already_present(prompt))


class SignCopyAcceptTests(unittest.TestCase):
    def test_accept_no_change_protocol(self) -> None:
        inp = "a taco walks under neon glow"
        out = "NO_CHANGE\n\n" + inp
        self.assertEqual(accept_sign_copy_output(inp, out), inp)

    def test_accept_updated_with_new_quotes(self) -> None:
        inp = "failing neon signs along the alley"
        out = 'UPDATED\n\nfailing neon signs reading "LAST CALL" along the alley'
        result = accept_sign_copy_output(inp, out)
        self.assertIn("LAST CALL", result)

    def test_reject_paraphrase_without_new_quotes(self) -> None:
        inp = "failing neon signs along the alley"
        out = "UPDATED\n\ndim neon signage lines the alleyway with more atmospheric detail here"
        self.assertEqual(accept_sign_copy_output(inp, out), inp)

    def test_reject_truncated(self) -> None:
        inp = "x" * 400
        # Force cue so accept path considers quotes
        inp = "failing neon signs " + inp
        out = "UPDATED\n\nshort"
        self.assertEqual(accept_sign_copy_output(inp, out), inp)

    def test_parse_protocol(self) -> None:
        status, body = parse_sign_copy_protocol('UPDATED\n\nhello "WORLD" there')
        self.assertEqual(status, "UPDATED")
        self.assertIn("WORLD", body)


class SignCopyRunTests(unittest.TestCase):
    def test_skips_when_no_cues(self) -> None:
        text, meta = run_sign_copy_pass("a quiet mountain lake at dawn", stage1_backend="comfy")
        self.assertEqual(text, "a quiet mountain lake at dawn")
        self.assertFalse(meta["ran"])
        self.assertEqual(meta["skipped_reason"], "no_cues")

    def test_skips_when_already_present(self) -> None:
        prompt = 'a neon sign reading "CLOSED"'
        text, meta = run_sign_copy_pass(prompt, stage1_backend="openrouter")
        self.assertEqual(text, prompt)
        self.assertEqual(meta["skipped_reason"], "already_present")

    def test_calls_comfy_with_free_vram_false_after_comfy_stage1(self) -> None:
        inp = "failing neon signs along the alley under fog"
        raw = 'UPDATED\n\nfailing neon signs reading "LAST BUS" along the alley under fog'

        with patch("comfy_qwen_vl.comfy_qwen_vl_available", return_value=True), patch(
            "comfy_qwen_vl.expand_prompt_comfy", return_value=raw
        ) as expand:
            text, meta = run_sign_copy_pass(inp, stage1_backend="comfy")
        self.assertIn("LAST BUS", text)
        self.assertTrue(meta["ran"])
        self.assertTrue(meta["changed"])
        self.assertFalse(expand.call_args.kwargs.get("free_vram", True))

    def test_calls_comfy_with_free_vram_true_after_openrouter(self) -> None:
        inp = "a woman holds a handwritten paper toward the camera in a dim room"
        raw = 'UPDATED\n\na woman holds a handwritten paper reading "COME HOME" toward the camera in a dim room'

        with patch("comfy_qwen_vl.comfy_qwen_vl_available", return_value=True), patch(
            "comfy_qwen_vl.expand_prompt_comfy", return_value=raw
        ) as expand:
            text, meta = run_sign_copy_pass(inp, stage1_backend="openrouter")
        self.assertIn("COME HOME", text)
        self.assertTrue(expand.call_args.kwargs.get("free_vram", False))

    def test_comfy_error_keeps_stage1(self) -> None:
        inp = "failing neon signs along the alley"
        with patch("comfy_qwen_vl.comfy_qwen_vl_available", return_value=True), patch(
            "comfy_qwen_vl.expand_prompt_comfy", side_effect=RuntimeError("boom")
        ):
            text, meta = run_sign_copy_pass(inp, stage1_backend="comfy")
        self.assertEqual(text, inp)
        self.assertEqual(meta["skipped_reason"], "error")

    def test_splice_helper(self) -> None:
        from sign_copy_pass import splice_copy_into_prompt

        out = splice_copy_into_prompt("failing neon signs along the alley", "LAST CALL")
        self.assertIn('reading "LAST CALL"', out)

    def test_micro_invent_fallback_when_rewrite_fails(self) -> None:
        inp = "failing neon signs along the alley under fog"
        # Full rewrite returns UPDATED without quotes; micro-invent returns a phrase.
        responses = [
            "UPDATED\n\nfailing neon signs along the alley under fog",
            "UPDATED\n\nfailing neon signs along the alley under fog",
            '"GHOST TOWN"',
        ]

        def fake_expand(prompt, system_prompt, **kwargs):
            return responses.pop(0)

        with patch("comfy_qwen_vl.comfy_qwen_vl_available", return_value=True), patch(
            "comfy_qwen_vl.expand_prompt_comfy", side_effect=fake_expand
        ):
            text, meta = run_sign_copy_pass(inp, stage1_backend="comfy")
        self.assertIn("GHOST TOWN", text)
        self.assertTrue(meta["changed"])
        self.assertEqual(meta.get("fallback"), "micro_invent_splice")


if __name__ == "__main__":
    unittest.main()
