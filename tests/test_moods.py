from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moods import MOODS, apply_mood


class MoodTests(unittest.TestCase):
    def test_all_horror_moods_are_photography_based(self) -> None:
        horror = [m for m in MOODS if m["category"] == "Horror"]
        self.assertGreater(len(horror), 0)
        for mood in horror:
            text = f"{mood['keywords']} {mood['avoids']}".lower()
            self.assertIn("photorealistic", text, mood["id"])
            self.assertIn("photography", text, mood["id"])
            self.assertIn("film grain", text, mood["id"])
            self.assertIn("illustration", mood["avoids"].lower(), mood["id"])

    def test_apply_mood_supports_style_mashups(self) -> None:
        prompt, negative = apply_mood("a lonely roadside motel", "", "gothic_horror,vintage_35mm,film_noir")

        self.assertIn("gothic horror", prompt)
        self.assertIn("vintage 35mm", prompt)
        self.assertIn("film noir", prompt)
        self.assertIn("a lonely roadside motel", prompt)
        self.assertIn("illustration", negative)
        self.assertIn("digital clarity", negative)


if __name__ == "__main__":
    unittest.main()
