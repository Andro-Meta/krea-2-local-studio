from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from prompt_expander import (
    _decode_generation,
    describe_image_local,
    describe_image_openrouter,
    expand_prompt_local,
    expand_prompt_result,
    ideogram_json_to_krea_prompt,
)


class PromptExpanderTests(unittest.TestCase):
    def test_reports_local_helper_failure(self) -> None:
        with (
            patch("prompt_expander._load_local_qwen", side_effect=RuntimeError("missing local model")),
            patch("prompt_expander.logger.warning"),
        ):
            result = expand_prompt_result("a haunted house", backend="local")

        self.assertEqual(result.expanded, "a haunted house")
        self.assertFalse(result.changed)
        self.assertIsNotNone(result.error)

    def test_local_qwen_expands_prompt(self) -> None:
        class FakeTensor:
            def to(self, *_args, **_kwargs):
                return self

        class FakeTokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
                self.messages = messages
                return FakeTensor()

            def decode(self, tokens, skip_special_tokens=True):
                return "A cinematic local expanded prompt."

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                return [[1, 2, 3]]

        with patch("prompt_expander._load_local_qwen", return_value=(FakeTokenizer(), None, FakeModel())):
            result = expand_prompt_local("a small chapel")

        self.assertTrue(result.changed)
        self.assertEqual(result.backend, "local")
        self.assertIn("cinematic", result.expanded)

    def test_local_qwen_expands_prompt_with_batch_encoding(self) -> None:
        class FakeInputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        class FakeTokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
                return FakeInputs({"input_ids": [10, 11], "attention_mask": [1, 1]})

            def decode(self, tokens, skip_special_tokens=True):
                return "A v5-compatible local expanded prompt."

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                self.kwargs = kwargs
                assert "input_ids" in kwargs
                assert "attention_mask" in kwargs
                return [[10, 11, 12, 13]]

        model = FakeModel()
        with patch("prompt_expander._load_local_qwen", return_value=(FakeTokenizer(), None, model)):
            result = expand_prompt_local("a small chapel")

        self.assertTrue(result.changed)
        self.assertEqual(result.backend, "local")
        self.assertIn("v5-compatible", result.expanded)

    def test_decode_generation_accepts_tensor_like_outputs(self) -> None:
        class FakeInput:
            shape = [1, 2]

        class FakeOutputs:
            def __bool__(self):
                raise RuntimeError("tensor truth value is ambiguous")

            def __getitem__(self, index):
                self.index = index
                return [10, 11, 12, 13]

        class FakeTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return "decoded " + ",".join(str(token) for token in tokens)

        text = _decode_generation(FakeTokenizer(), FakeOutputs(), FakeInput())

        self.assertEqual(text, "decoded 12,13")

    def test_local_qwen_describes_image(self) -> None:
        class FakeInputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        class FakeTokenizer:
            eos_token_id = 9

            def decode(self, tokens, skip_special_tokens=True):
                return "A moody local image prompt."

        class FakeProcessor:
            def __call__(self, **_kwargs):
                return FakeInputs()

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                return [[1, 2, 3]]

        with patch("prompt_expander._load_local_qwen", return_value=(FakeTokenizer(), FakeProcessor(), FakeModel())):
            tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
            result = describe_image_local(tiny_png)

        self.assertEqual(result["backend"], "local")
        self.assertIn("local image", result["prompt"])

    def test_openrouter_expands_with_free_fallback_models(self) -> None:
        calls = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "A cinematic haunted house prompt."}}]}

        def fake_post(url: str, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

        with patch("prompt_expander.requests.post", side_effect=fake_post):
            result = expand_prompt_result(
                "a haunted house",
                backend="openrouter",
                openrouter_api_key="test-openrouter-key",
                openrouter_model="anthropic/claude-3.5-sonnet",
                openrouter_free_only=True,
            )

        self.assertTrue(result.changed)
        self.assertEqual(result.backend, "openrouter")
        payload = calls[0][1]["json"]
        self.assertTrue(payload["model"].endswith(":free"))
        self.assertIn("models", payload)
        self.assertLessEqual(len(payload["models"]), 3)
        self.assertEqual(calls[0][1]["headers"]["Authorization"], "Bearer test-openrouter-key")

    def test_openrouter_rate_limit_returns_hard_error(self) -> None:
        class OpenRouterResponse:
            def raise_for_status(self) -> None:
                raise Exception("429 Too Many Requests")

            def json(self) -> dict:
                return {}

        def fake_post(url: str, **_kwargs):
            return OpenRouterResponse()

        with (
            patch("prompt_expander.requests.post", side_effect=fake_post),
            patch("prompt_expander.logger.warning"),
        ):
            result = expand_prompt_result(
                "a haunted house",
                backend="openrouter",
                openrouter_api_key="test-openrouter-key",
            )

        self.assertFalse(result.changed)
        self.assertEqual(result.backend, "openrouter")
        self.assertIn("rate limit", result.error.lower())

    def test_ideogram_json_is_flattened_for_krea(self) -> None:
        prompt = ideogram_json_to_krea_prompt({
            "high_level_description": "A haunted farmhouse under a stormy sky",
            "style_description": {
                "aesthetics": "photorealistic, gritty",
                "lighting": "low moonlight",
                "photo": "35mm film grain",
                "medium": "photograph",
            },
            "compositional_deconstruction": {
                "background": "Rows of dead corn fade into fog.",
                "elements": [
                    {"type": "obj", "desc": "a lone figure in a raincoat"},
                    {"type": "text", "text": "BEWARE", "desc": "weathered sign text"},
                ],
            },
        })

        self.assertIn("haunted farmhouse", prompt)
        self.assertIn("35mm film grain", prompt)
        self.assertIn("Rows of dead corn", prompt)
        self.assertIn("lone figure", prompt)

    def test_ideogram_backend_calls_magic_prompt_api(self) -> None:
        calls = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "json_prompt": {
                        "high_level_description": "A cinematic haunted house",
                        "compositional_deconstruction": {"background": "Foggy night", "elements": []},
                    }
                }

        def fake_post(url: str, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

        with patch("prompt_expander.requests.post", side_effect=fake_post):
            result = expand_prompt_result(
                "haunted house",
                backend="ideogram-json",
                ideogram_api_key="ideo-test",
            )

        self.assertEqual(result.backend, "ideogram-json")
        self.assertTrue(result.changed)
        self.assertIn("ideogram-v4/magic-prompt", calls[0][0])
        self.assertEqual(calls[0][1]["headers"]["Api-Key"], "ideo-test")
        self.assertEqual(calls[0][1]["json"]["text_prompt"], "haunted house")

    def test_describe_image_uses_openrouter_vision_fallbacks(self) -> None:
        calls = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "A grainy photograph of a chapel."}}]}

        def fake_post(url: str, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

        with patch("prompt_expander.requests.post", side_effect=fake_post):
            result = describe_image_openrouter("ZmFrZQ==", "test-openrouter-key")

        self.assertEqual(result["prompt"], "A grainy photograph of a chapel.")
        self.assertEqual(result["backend"], "openrouter")
        payload = calls[0][1]["json"]
        self.assertIn("models", payload)
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
