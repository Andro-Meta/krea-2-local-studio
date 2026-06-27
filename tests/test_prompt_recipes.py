from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class PromptRecipeTests(unittest.TestCase):
    def test_save_list_and_delete_recipe(self) -> None:
        from prompt_recipes import delete_recipe, list_recipes, save_recipe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.json"
            recipe = save_recipe(
                {
                    "name": "Neon fashion recipe",
                    "prompt": "neon jacket editorial",
                    "loras": [{"name": "style", "strength": 0.7}],
                    "moodboard_uuids": ["abc"],
                    "seed_variance_preset": "balanced",
                },
                path=path,
            )

            self.assertEqual(recipe["id"], "neon-fashion-recipe")
            self.assertEqual(list_recipes(path=path)[0]["loras"][0]["name"], "style")
            self.assertTrue(delete_recipe(recipe["id"], path=path))
            self.assertEqual(list_recipes(path=path), [])

    def test_save_recipe_updates_existing_id(self) -> None:
        from prompt_recipes import list_recipes, save_recipe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.json"
            save_recipe({"id": "cinematic", "name": "Cinematic", "prompt": "one"}, path=path)
            save_recipe({"id": "cinematic", "name": "Cinematic", "prompt": "two"}, path=path)

            recipes = list_recipes(path=path)
            self.assertEqual(len(recipes), 1)
            self.assertEqual(recipes[0]["prompt"], "two")


if __name__ == "__main__":
    unittest.main()
