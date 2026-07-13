from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class AnimateFrontendContractTests(unittest.TestCase):
    def test_generated_contract_matches_backend_exactly(self) -> None:
        from export_animate_contract import build_contract

        artifact = (
            ROOT / "frontend" / "src" / "generated" / "animate-contract.json"
        )
        committed = json.loads(artifact.read_text(encoding="utf-8"))

        self.assertEqual(committed, build_contract())

    def test_contract_contains_bounds_endpoints_result_and_task_kind(self) -> None:
        from export_animate_contract import build_contract

        contract = build_contract()
        properties = contract["request_schema"]["properties"]
        self.assertEqual(properties["render_frames"]["anyOf"][0]["maximum"], 720)
        self.assertEqual(properties["width"]["multipleOf"], 16)
        self.assertEqual(properties["steps"]["minimum"], 3)
        self.assertEqual(contract["endpoints"]["submit"], "/api/animate")
        self.assertEqual(contract["task_kind"], "animation")
        self.assertEqual(
            list(contract["result_schema"]["properties"]),
            contract["result_fields"],
        )
        self.assertEqual(
            contract["result_fields"],
            ["video_url", "poster_url", "frame_count", "fps", "duration", "gallery_id"],
        )


if __name__ == "__main__":
    unittest.main()
