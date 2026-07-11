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
                    "mood": "retro_web,film_noir",
                    "moodboard_strength": 0.65,
                    "moodboard_uuids": ["abc"],
                    "seed_variance_preset": "balanced",
                },
                path=path,
            )

            self.assertEqual(recipe["id"], "neon-fashion-recipe")
            self.assertEqual(list_recipes(path=path)[0]["loras"][0]["name"], "style")
            self.assertEqual(list_recipes(path=path)[0]["mood"], "retro_web,film_noir")
            self.assertEqual(list_recipes(path=path)[0]["moodboard_strength"], 0.65)
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


class PerUserRecipeTests(unittest.TestCase):
    def test_users_see_own_and_legacy_shared_recipes_only(self) -> None:
        from prompt_recipes import list_recipes, save_recipe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.json"
            save_recipe({"id": "shared-legacy", "name": "Legacy", "prompt": "x"}, path=path)  # no owner
            save_recipe({"id": "alices", "name": "Alices", "prompt": "a"}, path=path, username="alice")
            save_recipe({"id": "bobs", "name": "Bobs", "prompt": "b"}, path=path, username="bob")

            alice_ids = {r["id"] for r in list_recipes(path=path, username="alice")}
            self.assertEqual(alice_ids, {"shared-legacy", "alices"})
            # Local mode (no username) sees everything.
            all_ids = {r["id"] for r in list_recipes(path=path)}
            self.assertEqual(all_ids, {"shared-legacy", "alices", "bobs"})

    def test_same_id_is_scoped_per_user(self) -> None:
        from prompt_recipes import list_recipes, save_recipe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.json"
            save_recipe({"id": "cinematic", "name": "Cinematic", "prompt": "alices version"}, path=path, username="alice")
            save_recipe({"id": "cinematic", "name": "Cinematic", "prompt": "bobs version"}, path=path, username="bob")

            alice = list_recipes(path=path, username="alice")
            bob = list_recipes(path=path, username="bob")
            self.assertEqual([r["prompt"] for r in alice], ["alices version"])
            self.assertEqual([r["prompt"] for r in bob], ["bobs version"])

    def test_delete_respects_ownership(self) -> None:
        from prompt_recipes import delete_recipe, list_recipes, save_recipe

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.json"
            save_recipe({"id": "shared-legacy", "name": "Legacy", "prompt": "x"}, path=path)
            save_recipe({"id": "bobs", "name": "Bobs", "prompt": "b"}, path=path, username="bob")

            # Alice cannot delete Bob's recipe or the shared legacy one.
            self.assertFalse(delete_recipe("bobs", path=path, username="alice"))
            self.assertFalse(delete_recipe("shared-legacy", path=path, username="alice"))
            # Bob deletes his own; admin deletes the legacy shared one.
            self.assertTrue(delete_recipe("bobs", path=path, username="bob"))
            self.assertTrue(delete_recipe("shared-legacy", path=path, username="root", is_admin=True))
            self.assertEqual(list_recipes(path=path), [])


if __name__ == "__main__":
    unittest.main()
