"""Official Krea 2 prompting guidance.

Distilled from krea-ai/krea-2 docs/prompting.md so Studio can surface the same
guidance the Krea team recommends, and so the prompt planner follows it.
"""
from __future__ import annotations

OFFICIAL_PROMPTING_GUIDE = (
    "Krea 2 prompting (official guidelines):\n"
    "- Use natural language prompts; describe the scene as you would to a person.\n"
    "- The Turbo model can generate up to 2k resolution images.\n"
    "- Long, detailed prompts yield the best results, but the model also does well "
    "with minimal prompt engineering.\n"
    "- For text rendering, put the words to be rendered in quotes (e.g. a sign that "
    "says \"OPEN\").\n"
    "- For LLM-assisted expansion, use the official expansion prompt as the system "
    "prompt (Studio's prompt planner/expander already does)."
)

# A few representative official example prompts (natural-language, detailed style).
OFFICIAL_PROMPT_EXAMPLES = [
    "immense rocket launch exhaust as seen from extremely close up",
    (
        "close-up anime portrait of a young woman, large amber-brown eyes with intricate "
        "sparkling reflections, index finger delicately touching a subtle smile, messy dark "
        "blue hair with loose strands crossing her face, white and navy school uniform, bright "
        "high-key lighting, detailed digital painting, shallow depth of field"
    ),
    (
        "A tiny, russet-brown harvest mouse clings to a slender diagonal branch amid vibrant "
        "green lobed leaves; macro photograph, extremely shallow depth of field on the face, "
        "creamy green bokeh, soft diffused natural lighting"
    ),
    (
        "high-fashion editorial portrait of a young woman, short platinum blonde bob with heavy "
        "bangs, structured black top, gold hoop earrings, solid crimson red background, soft "
        "directional studio lighting, cinematic color palette, medium close-up"
    ),
]


def prompting_guide_payload() -> dict:
    return {
        "guidelines": OFFICIAL_PROMPTING_GUIDE,
        "examples": list(OFFICIAL_PROMPT_EXAMPLES),
        "source": "krea-ai/krea-2 docs/prompting.md",
    }
