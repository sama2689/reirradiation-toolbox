from __future__ import annotations

"""Editable default settings for DVH evaluation.

Update this file when you want to change which structures are evaluated,
what alpha/beta value is used for each structure, or which DVH metrics are
reported in the dose evaluation tab.
"""

from typing import Any

# This is intentionally a simple list of dictionaries so it is easy to edit.
# Add, remove, or adjust entries here as needed.
DEFAULT_DVH_SETTINGS: list[dict[str, Any]] = [
    {
        "region": "cord",
        "alpha_beta": 2.0,
        "metrics": ["D50", "Dmax", "D1cc"],
    },
    {
        "region": "brain",
        "alpha_beta": 2.0,
        "metrics": ["D50", "Dmax"],
    },
    {
        "region": "body",
        "alpha_beta": 3.0,
        "metrics": ["D50", "Dmax"],
    },
]


def default_dvh_settings() -> list[dict[str, Any]]:
    """Return a copy of the default DVH settings list."""
    return [dict(entry) for entry in DEFAULT_DVH_SETTINGS]


def dvh_setting_for_region(region: str) -> dict[str, Any] | None:
    """Return the default DVH setting for a region name, if present."""
    target = region.strip().lower()
    for entry in DEFAULT_DVH_SETTINGS:
        if str(entry.get("region", "")).strip().lower() == target:
            return dict(entry)
    return None


DEFAULT_FALLBACK_DVH_SETTING: dict[str, Any] = {
    "alpha_beta": 3.0,
    "metrics": ["D50"],
}


def default_dvh_setting_for_region(region: str) -> dict[str, Any]:
    """Return an editable default setting for a region, falling back to a generic entry."""
    setting = dvh_setting_for_region(region)
    if setting is not None:
        return setting
    return {"region": str(region).strip().lower(), **dict(DEFAULT_FALLBACK_DVH_SETTING)}


def default_dvh_settings_for_regions(regions: list[str]) -> list[dict[str, Any]]:
    """Return settings for the provided regions, adding a fallback entry for any missing contour."""
    seen: set[str] = set()
    settings: list[dict[str, Any]] = []
    for entry in DEFAULT_DVH_SETTINGS:
        region = str(entry.get("region", "")).strip().lower()
        if region and region in {str(r).strip().lower() for r in regions}:
            settings.append(dict(entry))
            seen.add(region)
    for region in regions:
        normalized = str(region).strip().lower()
        if normalized and normalized not in seen:
            settings.append(default_dvh_setting_for_region(normalized))
            seen.add(normalized)
    return settings
