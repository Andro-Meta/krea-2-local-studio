from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ModerationTests(unittest.TestCase):
    def test_prompt_policy_blocks_explicit_child_prompt(self) -> None:
        from moderation import moderate_prompt

        decision = moderate_prompt("photorealistic nude woman in bed", role="child")

        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.event_type, "prompt")
        self.assertIn("explicit", decision.reason.lower())
        self.assertGreater(decision.scores["policy_score"], 0.0)

    def test_prompt_policy_allows_safe_child_prompt(self) -> None:
        from moderation import moderate_prompt

        decision = moderate_prompt("a watercolor dragon reading a book", role="child")

        self.assertEqual(decision.action, "allow")
        self.assertEqual(decision.event_type, "prompt")

    def test_prompt_policy_does_not_block_admin_or_user(self) -> None:
        from moderation import moderate_prompt

        for role in ("admin", "user"):
            with self.subTest(role=role):
                decision = moderate_prompt("photorealistic nude figure study", role=role)
                self.assertEqual(decision.action, "allow")

    def test_image_policy_blocks_explicit_detector_hit_for_child(self) -> None:
        from PIL import Image

        from moderation import moderate_image

        class FakeProvider:
            def detect(self, _image):
                return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.92}]

        decision = moderate_image(Image.new("RGB", (8, 8)), role="child", provider=FakeProvider())

        self.assertEqual(decision.action, "block")
        self.assertEqual(decision.event_type, "image")
        self.assertIn("FEMALE_BREAST_EXPOSED", decision.labels)
        self.assertGreater(decision.scores["explicit_score"], 0.9)

    def test_image_policy_fails_closed_when_provider_missing_for_child(self) -> None:
        from PIL import Image

        from moderation import moderate_image

        decision = moderate_image(Image.new("RGB", (8, 8)), role="child", provider=None)

        self.assertEqual(decision.action, "block")
        self.assertIn("provider unavailable", decision.reason)

    def test_moderation_events_are_persisted(self) -> None:
        from moderation import init_moderation_db, list_moderation_events, save_moderation_event

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"

            async def run() -> None:
                await init_moderation_db(db_path)
                event_id = await save_moderation_event(
                    db_path=db_path,
                    username="kid1",
                    role="child",
                    event_type="prompt",
                    action="block_prompt",
                    prompt="bad prompt",
                    negative_prompt="",
                    mode="txt2img",
                    scores={"policy_score": 1.0},
                    reason="explicit sexual term",
                    job_id="job-1",
                )
                data = await list_moderation_events(db_path=db_path, limit=10)
                self.assertEqual(event_id, 1)
                self.assertEqual(data["total"], 1)
                self.assertEqual(data["items"][0]["username"], "kid1")
                self.assertEqual(data["items"][0]["action"], "block_prompt")
                self.assertEqual(data["items"][0]["scores"]["policy_score"], 1.0)

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
