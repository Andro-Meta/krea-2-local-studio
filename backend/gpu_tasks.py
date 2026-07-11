GENERATION = "generation"
PROMPT_EXPAND = "prompt_expand"
PROMPT_PLAN = "prompt_plan"
IMAGE_DESCRIBE = "image_describe"
UPSCALE = "upscale"
DEPTH_PREVIEW = "depth_preview"
MOODBOARD_GUIDANCE = "moodboard_guidance"
BACKGROUND_ENRICHMENT = "background_enrichment"
MODEL_WARMUP = "model_warmup"
HELPER_BENCHMARK = "helper_benchmark"

HELPER_KINDS = frozenset(
    {
        PROMPT_EXPAND,
        PROMPT_PLAN,
        IMAGE_DESCRIBE,
        UPSCALE,
        DEPTH_PREVIEW,
        MOODBOARD_GUIDANCE,
        BACKGROUND_ENRICHMENT,
        MODEL_WARMUP,
        HELPER_BENCHMARK,
    }
)

DISPLAY_LABELS = {
    GENERATION: "Generation",
    PROMPT_EXPAND: "Prompt expansion",
    PROMPT_PLAN: "Prompt planning",
    IMAGE_DESCRIBE: "Image description",
    UPSCALE: "Upscale",
    DEPTH_PREVIEW: "Depth preview",
    MOODBOARD_GUIDANCE: "Moodboard guidance",
    BACKGROUND_ENRICHMENT: "Background enrichment",
    MODEL_WARMUP: "Model warmup",
    HELPER_BENCHMARK: "Helper benchmark",
}


def foreign_summary(task_kind: str) -> str:
    if task_kind in HELPER_KINDS:
        return "Another user's helper"
    return "Another user's generation"
