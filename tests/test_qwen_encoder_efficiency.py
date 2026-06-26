from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.encoder import Qwen3VLConditioner


class QwenEncoderEfficiencyTests(unittest.TestCase):
    def test_vision_processor_is_cached(self) -> None:
        conditioner = object.__new__(Qwen3VLConditioner)
        conditioner._version = "local-qwen"
        conditioner._vision_processor = None
        processor = object()

        with patch("transformers.Qwen3VLProcessor.from_pretrained", return_value=processor) as load:
            self.assertIs(conditioner._get_vision_processor(), processor)
            self.assertIs(conditioner._get_vision_processor(), processor)

        load.assert_called_once_with("local-qwen")

    def test_duplicate_text_prompts_encode_once_then_expand(self) -> None:
        conditioner = object.__new__(Qwen3VLConditioner)
        calls = []

        def fake_encode_unique(self, prompts):
            calls.append(list(prompts))
            return (
                torch.ones(1, 2, 3, 4, dtype=torch.bfloat16),
                torch.tensor([[True, False]]),
            )

        conditioner._encode_unique_text = types.MethodType(fake_encode_unique, conditioner)

        txt, mask = Qwen3VLConditioner.forward(conditioner, ["same prompt"] * 4)

        self.assertEqual(calls, [["same prompt"]])
        self.assertEqual(tuple(txt.shape), (4, 2, 3, 4))
        self.assertEqual(tuple(mask.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
