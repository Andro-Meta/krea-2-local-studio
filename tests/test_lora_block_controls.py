from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from lora_manager import _block_filter_allows, _lora_module_group  # noqa: E402


class LoraBlockControlTests(unittest.TestCase):
    def test_module_names_map_to_semantic_groups(self) -> None:
        self.assertEqual(_lora_module_group("blocks.0.attn.wq"), "early")
        self.assertEqual(_lora_module_group("blocks.14.attn.wq"), "middle")
        self.assertEqual(_lora_module_group("blocks.27.attn.wq"), "late")
        self.assertEqual(_lora_module_group("txtfusion.projector"), "style")

    def test_style_safe_filter_keeps_style_and_late_blocks(self) -> None:
        self.assertTrue(_block_filter_allows("txtfusion.projector", "style_safe"))
        self.assertTrue(_block_filter_allows("blocks.24.attn.wq", "style_safe"))
        self.assertFalse(_block_filter_allows("blocks.1.attn.wq", "style_safe"))

    def test_unknown_filter_skips_safely(self) -> None:
        self.assertFalse(_block_filter_allows("blocks.1.attn.wq", "does_not_exist"))


if __name__ == "__main__":
    unittest.main()
