from __future__ import annotations

from dataclasses import dataclass


STRICT_EDIT_MODES = {"inpaint", "outpaint"}


@dataclass(frozen=True)
class EditProviderChoice:
    name: str
    reason: str
    strict: bool = False


def resolve_edit_provider(
    requested: str | None,
    mode: str,
    *,
    flux_fill_installed: bool = False,
) -> EditProviderChoice:
    requested = (requested or "auto").strip().lower()
    mode = (mode or "").strip().lower()

    if requested in {"krea", "krea_native"}:
        return EditProviderChoice("krea_native", "Using Krea native edit pipeline.")

    if requested == "flux_fill":
        if not flux_fill_installed:
            return EditProviderChoice(
                "krea_native",
                "FLUX Fill precision editing is not installed; using Krea native fallback.",
            )
        return EditProviderChoice("flux_fill", "Using FLUX Fill precision edit provider.", strict=True)

    if mode in STRICT_EDIT_MODES and flux_fill_installed:
        return EditProviderChoice("flux_fill", "Strict edit mode with FLUX Fill installed.", strict=True)

    if mode in STRICT_EDIT_MODES:
        return EditProviderChoice(
            "krea_native",
            "FLUX Fill precision editing is not installed; using Krea native fallback.",
        )

    return EditProviderChoice("krea_native", "Creative Krea redraw uses the Krea native provider.")
