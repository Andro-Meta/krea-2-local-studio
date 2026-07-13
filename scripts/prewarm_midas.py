from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

MARKER_NAME = "krea-midas-small-ready.json"


def prewarm(marker: Path) -> dict[str, object]:
    import torch

    # MiDaS_small loads this legacy EfficientNet repository internally without
    # forwarding trust_repo. Trust and cache the exact dependency first so an
    # unattended install never blocks on Torch Hub's interactive prompt.
    dependency = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=False,
        trust_repo=True,
    )
    del dependency
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to("cpu").eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if not hasattr(transforms, "small_transform"):
        raise RuntimeError("MiDaS transforms cache does not provide small_transform")

    hub_dir = Path(torch.hub.get_dir()).resolve()
    weights = hub_dir / "checkpoints" / "midas_v21_small_256.pt"
    repositories = sorted(
        path for pattern in ("intel-isl_MiDaS*", "isl-org_MiDaS*")
        for path in hub_dir.glob(pattern)
        if path.is_dir()
    )
    if not weights.is_file() or not repositories:
        raise RuntimeError("MiDaS_small loaded but expected torch.hub cache is incomplete")

    payload = {
        "version": 1,
        "model": "MiDaS_small",
        "weights_path": str(weights),
        "hub_repo_path": str(repositories[-1]),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=marker.parent,
        prefix=f".{marker.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, marker)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if args.marker.name != MARKER_NAME:
        parser.error(f"marker filename must be {MARKER_NAME}")
    prewarm(args.marker.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
