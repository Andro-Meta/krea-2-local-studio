from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTests(unittest.TestCase):
    def test_run_bat_escapes_strftime_percent_tokens(self) -> None:
        text = (ROOT / "run.bat").read_text(encoding="utf-8")
        self.assertIn("strftime('%%Y%%m%%d-%%H%%M%%S')", text)
        self.assertNotIn("strftime('%Y%m%d-%H%M%S')", text)

    def test_comfy_startup_reports_effective_mode_and_pid(self) -> None:
        text = (ROOT / "scripts" / "start_comfyui.ps1").read_text(encoding="utf-8")
        self.assertIn("Effective ComfyUI:", text)
        self.assertIn("$process.Id", text)
        self.assertIn("Set vram state to:", text)

    def test_example_uses_canonical_high_vram_setting(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("KREA_COMFY_VRAM_MODE=highvram", text)

    def test_animate_documentation_matches_pinned_runtime_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("https://github.com/Dream-Making-Git/KreaDeforum", readme)
        self.assertIn("49bb6752ab045fac25652f3e9207d4706bf5c646", readme)
        self.assertIn("krea2-chunking-v2", readme)
        self.assertIn("60 FPS controls playback/interpolation", readme)
        self.assertIn("Animate", example)
        self.assertIn("owner-scoped animation frames", example)


if __name__ == "__main__":
    unittest.main()
