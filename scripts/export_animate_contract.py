from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.gpu_tasks import ANIMATION  # noqa: E402
from backend.schemas import AnimateRequest, AnimationResult  # noqa: E402


ARTIFACT = ROOT / "frontend" / "src" / "generated" / "animate-contract.json"
RESULT_FIELDS = list(AnimationResult.model_fields)
ENDPOINTS = {
    "submit": "/api/animate",
    "upload": "/api/animate/uploads",
    "status": "/api/generate/{job_id}",
    "cancel": "/api/generate/{job_id}/cancel",
    "ack": "/api/generate/{job_id}/ack",
    "websocket": "/ws/{job_id}",
}


def build_contract() -> dict:
    return {
        "version": 1,
        "task_kind": ANIMATION,
        "request_schema": AnimateRequest.model_json_schema(),
        "result_schema": AnimationResult.model_json_schema(),
        "endpoints": ENDPOINTS,
        "result_fields": RESULT_FIELDS,
    }


def write_contract(path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    write_contract()
