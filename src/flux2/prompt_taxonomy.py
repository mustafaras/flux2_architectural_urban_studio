"""Architecture and urban prompt taxonomy helpers."""

from __future__ import annotations

from typing import Any

PROMPT_TEMPLATE_VERSION = "arch-urban-v1"

MASSING_DESCRIPTORS = [
    "low-rise block",
    "mid-rise perimeter",
    "high-rise tower",
    "tower-podium",
    "courtyard massing",
]

FACADE_VOCABULARY = [
    "fully glazed",
    "opaque masonry",
    "timber expression",
    "metal panel rhythm",
    "deep shading fins",
]

MOBILITY_PUBLIC_REALM = [
    "car-free core",
    "transit-priority corridor",
    "bike-priority network",
    "highway-dominant access",
    "civic plaza activation",
]

ATMOSPHERE_CONTROLS = [
    "clear daytime",
    "golden hour",
    "night lighting",
    "overcast soft light",
    "rainy mood",
]

MASSING_PRESET_PACKS: dict[str, dict[str, Any]] = {
    "perimeter block": {
        "descriptors": ["mid-rise perimeter"],
        "preview": "Continuous edge blocks defining streets and interior court edges.",
        "recommended_use": "Dense urban infill with active street frontage.",
    },
    "courtyard": {
        "descriptors": ["courtyard massing", "mid-rise perimeter"],
        "preview": "Central open court surrounded by built form for shared amenity.",
        "recommended_use": "Residential or mixed-use blocks prioritizing inner open space.",
    },
    "slab": {
        "descriptors": ["low-rise block"],
        "preview": "Linear slab volume with repeated module rhythm.",
        "recommended_use": "Site-efficient layouts with simple structural logic.",
    },
    "tower-podium": {
        "descriptors": ["tower-podium", "high-rise tower"],
        "preview": "High-rise vertical element over activated lower podium.",
        "recommended_use": "Transit nodes and high-density mixed-use districts.",
    },
    "hybrid": {
        "descriptors": ["tower-podium", "courtyard massing", "mid-rise perimeter"],
        "preview": "Combined perimeter + vertical accents for layered skyline response.",
        "recommended_use": "Complex urban blocks balancing density and permeability.",
    },
}

FACADE_STYLE_PACKS: dict[str, dict[str, Any]] = {
    "modern": {
        "vocabulary": ["fully glazed", "metal panel rhythm"],
        "material_defaults": {"glazing_ratio": 60, "texture_roughness": 0.25, "reflectance_mood": "balanced"},
    },
    "contextual": {
        "vocabulary": ["opaque masonry", "deep shading fins"],
        "material_defaults": {"glazing_ratio": 35, "texture_roughness": 0.45, "reflectance_mood": "muted"},
    },
    "parametric": {
        "vocabulary": ["metal panel rhythm", "deep shading fins"],
        "material_defaults": {"glazing_ratio": 55, "texture_roughness": 0.3, "reflectance_mood": "reflective"},
    },
    "low-carbon emphasis": {
        "vocabulary": ["timber expression", "opaque masonry"],
        "material_defaults": {"glazing_ratio": 40, "texture_roughness": 0.55, "reflectance_mood": "muted"},
    },
}

PROGRAM_OVERLAYS = ["residential", "commercial", "mixed-use"]

CAMERA_FAMILIES = ["aerial", "eye-level", "courtyard", "streetscape"]

TIME_OF_DAY_PRESETS = ["morning", "noon", "golden hour", "night"]
SEASONAL_PRESETS = ["summer", "winter", "leaf-on", "leaf-off"]
CLIMATE_MOOD_PROFILES = ["temperate", "arid", "cold", "tropical"]

CONTINUITY_PROFILES = ["none", "day-night", "seasonal"]


def lint_prompt_slots(
    *,
    massing: list[str],
    facade: list[str],
    mobility: list[str],
    atmosphere: list[str],
) -> list[str]:
    """Return lint warnings for contradictory architecture intent."""
    warnings: list[str] = []

    if "low-rise block" in massing and "high-rise tower" in massing:
        warnings.append("Massing conflict: low-rise and high-rise selected together.")

    if "fully glazed" in facade and "opaque masonry" in facade:
        warnings.append("Façade conflict: fully glazed and opaque masonry selected together.")

    if "car-free core" in mobility and "highway-dominant access" in mobility:
        warnings.append("Mobility conflict: car-free and highway-dominant intent selected together.")

    if "clear daytime" in atmosphere and "night lighting" in atmosphere:
        warnings.append("Atmosphere conflict: daytime and night lighting selected together.")

    return warnings


def assemble_prompt(
    *,
    free_text: str,
    massing: list[str],
    facade: list[str],
    mobility: list[str],
    atmosphere: list[str],
) -> str:
    """Build final generation prompt from taxonomy slots and free text."""
    parts: list[str] = []

    trimmed_free_text = free_text.strip()
    if trimmed_free_text:
        parts.append(trimmed_free_text)

    if massing:
        parts.append(f"massing: {', '.join(massing)}")
    if facade:
        parts.append(f"facade: {', '.join(facade)}")
    if mobility:
        parts.append(f"mobility/public realm: {', '.join(mobility)}")
    if atmosphere:
        parts.append(f"atmosphere: {', '.join(atmosphere)}")

    return "; ".join(parts)


def build_prompt_taxonomy_payload(
    *,
    free_text: str,
    massing: list[str],
    facade: list[str],
    mobility: list[str],
    atmosphere: list[str],
) -> dict[str, Any]:
    """Build prompt taxonomy payload including lint results."""
    lint_results = lint_prompt_slots(
        massing=massing,
        facade=facade,
        mobility=mobility,
        atmosphere=atmosphere,
    )

    assembled_prompt = assemble_prompt(
        free_text=free_text,
        massing=massing,
        facade=facade,
        mobility=mobility,
        atmosphere=atmosphere,
    )

    return {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "slots": {
            "massing_descriptors": massing,
            "facade_vocabulary": facade,
            "mobility_public_realm": mobility,
            "atmosphere_controls": atmosphere,
        },
        "lint_results": lint_results,
        "free_text": free_text,
        "assembled_prompt": assembled_prompt,
    }


def apply_massing_preset(preset_name: str) -> dict[str, Any]:
    """Return a massing preset pack payload, or empty fallback if unknown."""
    preset = MASSING_PRESET_PACKS.get(preset_name.lower())
    if not preset:
        return {"descriptors": [], "preview": "", "recommended_use": ""}
    return {
        "descriptors": list(preset.get("descriptors", [])),
        "preview": str(preset.get("preview", "")),
        "recommended_use": str(preset.get("recommended_use", "")),
    }


def validate_material_controls(
    glazing_ratio: int,
    texture_roughness: float,
    reflectance_mood: str,
) -> tuple[dict[str, Any], list[str]]:
    """Clamp and validate façade/material controls with safe defaults."""
    warnings: list[str] = []

    clamped_glazing = max(0, min(95, int(glazing_ratio)))
    if clamped_glazing != int(glazing_ratio):
        warnings.append("Glazing ratio adjusted to safe range (0–95).")

    clamped_roughness = max(0.0, min(1.0, float(texture_roughness)))
    if abs(clamped_roughness - float(texture_roughness)) > 1e-9:
        warnings.append("Texture roughness adjusted to safe range (0.0–1.0).")

    allowed_moods = {"muted", "balanced", "reflective"}
    normalized_mood = str(reflectance_mood).strip().lower() or "balanced"
    if normalized_mood not in allowed_moods:
        normalized_mood = "balanced"
        warnings.append("Reflectance mood reset to safe default (balanced).")

    return (
        {
            "glazing_ratio": clamped_glazing,
            "texture_roughness": round(clamped_roughness, 3),
            "reflectance_mood": normalized_mood,
        },
        warnings,
    )


def build_phase2_prompt_segments(controls: dict[str, Any]) -> list[str]:
    """Build prompt segments from Phase 2 controls."""
    segments: list[str] = []

    massing_preset = str(controls.get("massing_preset", "")).strip()
    if massing_preset:
        segments.append(f"massing preset: {massing_preset}")

    facade_pack = str(controls.get("facade_style_pack", "")).strip()
    if facade_pack:
        segments.append(f"facade style pack: {facade_pack}")

    material = controls.get("material_realism", {}) if isinstance(controls.get("material_realism"), dict) else {}
    if material:
        segments.append(
            "material realism: "
            f"glazing {material.get('glazing_ratio', '?')}%, "
            f"roughness {material.get('texture_roughness', '?')}, "
            f"reflectance {material.get('reflectance_mood', '?')}"
        )

    program_overlays = controls.get("program_overlays", []) if isinstance(controls.get("program_overlays"), list) else []
    if program_overlays:
        segments.append(f"program overlay: {', '.join(program_overlays)}")

    view_lock = controls.get("view_lock", {}) if isinstance(controls.get("view_lock"), dict) else {}
    if view_lock:
        segments.append(
            "view lock: "
            f"{view_lock.get('camera_family', 'unknown')} "
            f"{view_lock.get('lens_mm', '?')}mm "
            f"lock={view_lock.get('perspective_lock', False)}"
        )

    environment = controls.get("environment", {}) if isinstance(controls.get("environment"), dict) else {}
    if environment:
        segments.append(
            "environment: "
            f"{environment.get('time_of_day', '')}, "
            f"{environment.get('seasonal_profile', '')}, "
            f"{environment.get('climate_mood', '')}"
        )

    return [seg for seg in segments if seg.strip()]


def calculate_continuity_score(controls: dict[str, Any]) -> float:
    """Compute lightweight continuity score for render-set coherence (0-100)."""
    score = 40.0

    view_lock = controls.get("view_lock", {}) if isinstance(controls.get("view_lock"), dict) else {}
    if bool(view_lock.get("perspective_lock", False)):
        score += 20.0

    seed_group_id = str(controls.get("seed_group_id", "")).strip()
    if seed_group_id:
        score += 20.0

    continuity_profile = str(controls.get("continuity_profile", "none")).strip().lower()
    if continuity_profile in {"day-night", "seasonal"}:
        score += 10.0

    environment = controls.get("environment", {}) if isinstance(controls.get("environment"), dict) else {}
    if environment.get("time_of_day") and environment.get("seasonal_profile"):
        score += 10.0

    return round(max(0.0, min(100.0, score)), 1)
