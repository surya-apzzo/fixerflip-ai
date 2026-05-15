"""Image edit engine: OpenAI -> Image."""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
from openai import AsyncOpenAI

from app.core import redis_cache
from app.core.config import settings
from app.schemas import ImageEditResult

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_EDIT_PROMPT_BASE = _PROMPTS_DIR / "editing_image_visual.txt"
_EDIT_PROMPT_BROAD = _PROMPTS_DIR / "editing_image_constraints_broad.txt"
_EDIT_PROMPT_TARGETED = _PROMPTS_DIR / "editing_image_constraints_targeted.txt"
_EDIT_PROMPT_REFERENCE = _PROMPTS_DIR / "editing_image_constraints_reference.txt"
IMAGE_CACHE_DIR = Path(".cache/renovation_image_downloads")
IMAGE_CACHE_TTL_SECONDS = settings.REDIS_CACHE_TTL_SECONDS
_DEFAULT_IMAGE_EDIT_INSTRUCTION = (
    "Repair only visible damage/issues in this property photo. Preserve layout, walls, finishes, "
    "furniture, and all intact objects. Do not redesign or restage."
)

_NO_ISSUES_DETECTED_INSTRUCTION = (
    "No specific AI-detected issues were found for this image. "
    "Default to a preservation-first edit: keep the image nearly identical and "
    "change only clearly visible defects (if any). "
    "Do not redesign, restage, replace finishes, or make structural changes. "
    "Do not alter intact cabinets, counters, flooring, walls, ceilings, fixtures, "
    "appliances, furniture, or decor unless explicitly requested by the user."
)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_OPENAI_IMAGE_EDIT_RETRY_BACKOFF_SECONDS = 0.8

_GPT_IMAGE_EDIT_SIZES = {
    "1024x1024": 1.0,
    "1536x1024": 1536 / 1024,
    "1024x1536": 1024 / 1536,
}


def _select_gpt_image_edit_size(image_bytes: bytes) -> str:
    """
    Pick a fixed GPT-Image canvas whose aspect ratio best matches the source file.

    Using only ``auto`` can change framing; matching aspect reduces perceived crop
    and missing left/right content in wide living-room shots.
    """
    try:
        from PIL import Image
    except ImportError:
        return "auto"
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            w, h = im.convert("RGB").size
    except Exception:
        return "auto"
    if w < 32 or h < 32:
        return "auto"
    ratio = max(w / h, 0.01)
    return min(_GPT_IMAGE_EDIT_SIZES, key=lambda k: abs(math.log(ratio) - math.log(_GPT_IMAGE_EDIT_SIZES[k])))


def _image_edit_supports_input_fidelity(model: str) -> bool:
    """input_fidelity is supported on gpt-image-1 / 1.5 family, not on -mini."""
    m = (model or "").strip().lower()
    if "mini" in m:
        return False
    return "gpt-image" in m or "image-1" in m


def _is_retryable_openai_exception(exc: Exception) -> bool:
    retryable_names = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
    return exc.__class__.__name__ in retryable_names


async def _wait_for_retry_backoff(attempt: int) -> None:
    await asyncio.sleep(_OPENAI_IMAGE_EDIT_RETRY_BACKOFF_SECONDS * (2**attempt))


def _build_image_download_headers(image_url: str) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": _BROWSER_USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    override = (settings.IMAGE_DOWNLOAD_REFERER or "").strip()
    if override:
        headers["Referer"] = override
    else:
        parsed = urlparse(image_url)
        if parsed.scheme and parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return headers


def _build_proxy_image_urls(image_url: str) -> list[str]:
    template = (settings.IMAGE_DOWNLOAD_PROXY_TEMPLATE or "").strip()
    if not template:
        return []
    if "://" in image_url:
        raw = image_url.split("://", 1)[1]
    else:
        raw = image_url
    candidates: list[str] = []
    for encoded_value in (
        quote(raw, safe=""),
        quote(f"https://{raw}", safe=""),
    ):
        url = template.replace("{url_no_scheme_encoded}", encoded_value)
        if url not in candidates:
            candidates.append(url)
    return candidates


_ELEMENT_HINTS = {
    "flooring": "floors and floor finish",
    "paint": "interior wall paint or exterior siding/trim/facade paint",
    "lighting": "light fixtures and lighting look",
    "furniture": "furniture style and condition",
    "ceiling": "interior ceiling surface, texture, and finish",
    "cabinet": "kitchen or bathroom cabinetry fronts",
    "window": "window frames and glass panes",
    "stair": "staircase steps, railing, and finish",
    "door": "interior and exterior doors",
    "roof": "roof shingles and roofline finish",
    "siding": "exterior siding, cladding, and facade boards",
    "landscaping": "planting beds, shrubs, trees, lawn, and yard landscape",
    "driveway": "driveway surface and approach paving",
    "fence": "perimeter fencing and gates",
}

_EXTERIOR_FLOORING_HINT = (
    "exterior deck, patio, porch boards, pavers, or other outdoor walkable surface finish"
)
_INTERIOR_FLOORING_HINT = "interior floors and floor finish"

_ELEMENT_ACTION_HINTS = {
    "flooring": (
        "Flooring directive: visibly replace the existing floor/walkable-surface finish everywhere it appears "
        "in frame for this element. Keep camera angle and scene geometry identical; do not alter unselected elements."
    ),
    "paint": (
        "Paint directive: for interiors, update only the requested wall/ceiling paint surfaces. "
        "For exteriors, update only visible siding/trim/shutter/body paint while keeping other building elements coherent."
    ),
    "lighting": (
        "Lighting directive: update only visible light fixtures and lighting tone/intensity while preserving all cabinetry, "
        "walls, flooring, furniture, and room geometry."
    ),
    "furniture": (
        "Furniture directive: update only furniture look/style/finish of existing furniture in place. "
        "Do not add extra furniture pieces and do not alter architecture, cabinetry, walls, or flooring."
    ),
    "ceiling": (
        "Ceiling directive: update only the visible ceiling surface (texture, color, or finish). "
        "Do not alter walls, floors, cabinets, lighting fixtures, or room geometry unless explicitly requested."
    ),
    "roof": (
        "Roof directive: visibly update the roof covering material and color. Preserve the structural roofline and the rest of the exterior unchanged."
    ),
    "cabinet": (
        "Cabinet directive: update the look, color, and finish of existing cabinets. Do not change the layout or surrounding walls and floors."
    ),
    "window": (
        "Window directive: update window frames or styles. Keep the wall structure and surrounding exterior/interior untouched."
    ),
    "stair": (
        "Stair directive: visibly update the staircase treads, risers, and railing style. Leave surrounding walls and flooring intact."
    ),
    "door": (
        "Door directive: update the style, color, or material of visible doors. Do not alter the surrounding walls or structural openings."
    ),
    "siding": (
        "Siding directive: visibly update exterior wall cladding/siding material, color, or pattern. "
        "Preserve roofline, windows, doors, and yard layout unless those are explicitly selected too."
    ),
    "landscaping": (
        "Landscaping directive: update planting beds, shrubs, small trees, lawn, and decorative landscape visible in frame. "
        "Do not replace the entire house structure; keep the home footprint and hardscape unless explicitly requested."
    ),
    "driveway": (
        "Driveway directive: update only the visible driveway or approach paving (material, color, condition). "
        "Preserve the house, yard, and fencing unless explicitly selected."
    ),
    "fence": (
        "Fence directive: update only visible perimeter fencing or gates (material, color, style). "
        "Preserve the house, roof, landscaping, and driveway unless explicitly selected."
    ),
}

_STRICT_DAMAGE_REPAIR_ONLY_RULES = (
    "CRITICAL DAMAGE-REPAIR RULES: Keep the same room identity and object inventory. "
    "Do not add any new furniture, kitchen cabinets, countertops, appliances, decor, fixtures, wall art, "
    "plants, doors, windows, or architectural elements unless the user explicitly requested that exact addition. "
    "Do not stage or redesign the room. Remove damage/debris and repair damaged surfaces in place only."
)

_SURFACE_CONTINUITY_RULES = (
    "Surface continuity rule: when repairing a damaged wall or ceiling zone, finish the entire visibly affected "
    "surface section with continuous texture/color so it does not look patchy, half-painted, or unfinished. "
    "Blend repaired boundaries naturally into adjacent intact areas."
)


def _is_exterior_renovation(type_of_renovation: str) -> bool:
    return str(type_of_renovation or "").strip().lower() == "exterior"


def _element_description(element: str, *, exterior: bool) -> str:
    if element == "flooring":
        return _EXTERIOR_FLOORING_HINT if exterior else _INTERIOR_FLOORING_HINT
    return _ELEMENT_HINTS.get(element, element)


def _reference_style_element_hint(element: str, *, exterior: bool) -> str:
    """Generic reference transfer for any allowed renovation element (not flooring-specific)."""
    scope = _element_description(element, exterior=exterior)
    label = element.replace("_", " ")
    return (
        f"Reference style ({label}): copy material, color, texture, and finish cues from image 2 onto "
        f"every visible {scope} in image 1. The change must be clearly visible compared with the source."
    )


def _element_action_hint(
    element: str,
    *,
    exterior: bool,
    reference_attached: bool = False,
) -> str | None:
    base = _ELEMENT_ACTION_HINTS.get(element)
    if not base:
        return None
    if reference_attached:
        return f"{_reference_style_element_hint(element, exterior=exterior)} {base}"
    return base


def _is_generic_renovate_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    generic_phrases = (
        "renovate",
        "please renovate",
        "renovate this",
        "renovate this one",
        "fully renovate",
        "make it renovated",
    )
    if any(phrase in t for phrase in generic_phrases):
        return True
    return bool(re.fullmatch(r"(please\s+)?renovate(\s+it|\s+this|\s+this\s+one)?[.!]?", t))


def _append_issue_repair_directives(
    directive_lines: list[str],
    *,
    issues: list[str],
    elements: list[str],
    element_descriptions: list[str],
    base_instruction: str,
) -> None:
    directive_lines.append("AI-detected issues/damages to repair: " + ", ".join(issues) + ".")
    directive_lines.append(
        "Focus repairs on the above detected issues. Fix each one while preserving "
        "all undamaged areas of the image."
    )
    directive_lines.append(_STRICT_DAMAGE_REPAIR_ONLY_RULES)
    directive_lines.append(_SURFACE_CONTINUITY_RULES)
    if any("ceiling" in issue.lower() or "wall" in issue.lower() for issue in issues):
        directive_lines.append(
            "For damaged wall/ceiling areas: fully restore each damaged section to a coherent finished state "
            "within that section (no exposed broken layers, no partial repaint blocks)."
        )
    if elements:
        directive_lines.append(
            "After completing required damage repairs, apply user-requested upgrades only to: "
            + ", ".join(element_descriptions)
            + "."
        )
        directive_lines.append(
            "These requested upgrades are secondary to safety/repair scope and must not remove necessary remediation work."
        )
    if not elements and _is_generic_renovate_request(base_instruction):
        directive_lines.append(
            "Generic renovate request detected. Treat this as damage/issue remediation only."
        )
        directive_lines.append(
            "Do not redesign, restage, or modernize intact areas. Keep original layout, style, and non-damaged finishes."
        )


def _append_selected_element_directives(
    directive_lines: list[str],
    *,
    visual_type: str,
    elements: list[str],
    element_descriptions: list[str],
    type_of_renovation: str,
    reference_image_attached: bool = False,
) -> None:
    exterior = _is_exterior_renovation(type_of_renovation)
    if element_descriptions:
        directive_lines.append(
            "Selected elements to renovate: " + ", ".join(element_descriptions) + "."
        )
        directive_lines.append("Apply visible changes only to selected elements.")
        if exterior:
            directive_lines.append(
                "Do NOT modify unselected exterior features (roof, windows, doors, siding, trim paint, "
                "landscaping, driveway, fence) unless explicitly selected or required by the user instruction."
            )
            directive_lines.append(
                "At least one visible change must be made for each selected element while preserving "
                "the same building footprint, camera angle, and overall composition."
            )
        else:
            directive_lines.append(
                "Do NOT modify unselected elements (especially wall color, ceiling, flooring, cabinets, counters, appliances) unless explicitly selected."
            )
            directive_lines.append(
                "At least one visible change must be made for each selected element while preserving the original room layout."
            )
        for element in elements:
            action_hint = _element_action_hint(
                element,
                exterior=exterior,
                reference_attached=reference_image_attached,
            )
            if action_hint:
                directive_lines.append(action_hint)
        return
    if visual_type != "select_elements_to_renovate":
        return
    directive_lines.append(
        "No specific elements were selected. Use conservative damage-repair mode only."
    )
    if exterior:
        directive_lines.append(
            "Keep all intact exterior features unchanged (roof, siding, windows, doors, trim, deck/patio flooring, driveway, fence, landscaping) "
            "unless damage repair requires local fixes."
        )
    else:
        directive_lines.append(
            "Keep all intact property features unchanged (especially cabinets, counters, flooring, walls, ceilings, fixtures, appliances, furniture, and decor)."
        )


def _append_reference_directives(
    directive_lines: list[str],
    *,
    visual_type: str,
    reference_image_url: str,
    reference_image_attached: bool,
    element_descriptions: list[str],
    elements: list[str] | None = None,
    type_of_renovation: str = "",
) -> None:
    ref = (reference_image_url or "").strip()
    if not ref and not reference_image_attached:
        return
    if reference_image_attached:
        directive_lines.append(
            "INPUT IMAGES (order matters): Image 1 = property photo to edit (same camera, layout, and composition). "
            "Image 2 = user style/material reference for the selected renovation elements."
        )
        if element_descriptions:
            directive_lines.append(
                "Apply image 2 styling ONLY to these selected elements on image 1: "
                + ", ".join(element_descriptions)
                + ". Leave every unselected surface unchanged."
            )
        else:
            directive_lines.append(
                "Use image 2 as style guidance only where the user instruction requires a visible change."
            )
        directive_lines.append(
            "The edit must be clearly visible on the selected surfaces. Match image 2's surface appearance "
            "(color, pattern, sheen, grout/joint treatment) on image 1—not image 2's furniture, plants, doors, or sky."
        )
        selected = {e.strip().lower() for e in (elements or []) if e}
        if _is_exterior_renovation(type_of_renovation) and "flooring" in selected:
            directive_lines.append(
                "Scope hint (exterior flooring): include deck boards, stair treads, and patio/paver fields "
                "visible in image 1 unless the user note narrows scope."
            )
        return
    if visual_type == "upload_my_own_reference_photo" and ref:
        directive_lines.append(
            "Reference image was provided but could not be downloaded; apply conservative edits only."
        )
    elif ref:
        directive_lines.append(f"Reference image URL (not attached to model): {ref}")


def _append_user_note_directive(directive_lines: list[str], *, base_instruction: str) -> None:
    if base_instruction:
        return
    directive_lines.append("User note: " + _DEFAULT_IMAGE_EDIT_INSTRUCTION)


def _append_authoritative_user_edit_block(directive_lines: list[str], *, base_instruction: str) -> None:
    """
    Put the client's words near the top of the directive so images.edit attends to them before
    long generic repair text. End of build_instruction_for_edit adds a short reminder.
    """
    directive_lines.extend(
        [
            "=== USER IMAGE EDIT REQUEST (AUTHORITATIVE) ===",
            "User instruction: " + base_instruction,
            (
                "Authority rule: If the user names specific colors, materials, surfaces, fixtures, rooms, "
                "or scope, implement those visibly. Sections below about conservative repair apply mainly to "
                "areas the user did not address; they must not cancel explicit user requests."
            ),
        ]
    )


def build_instruction_for_edit(
    *,
    user_inputs: str,
    type_of_renovation: str,
    visual_type: str,
    desired_quality_level: str,
    visual_scope_hint: str,
    reference_image_url: str = "",
    renovation_elements: list[str] | None = None,
    detected_issues: list[str] | None = None,
) -> str:
    base_instruction = (user_inputs or "").strip()
    elements = [e.strip().lower() for e in (renovation_elements or []) if e and e.strip()]
    exterior = _is_exterior_renovation(type_of_renovation)
    element_descriptions = [_element_description(e, exterior=exterior) for e in elements]
    ref_url = (reference_image_url or "").strip()
    reference_image_attached = bool(ref_url)

    if reference_image_attached:
        directive_lines = [
            "REFERENCE STYLE EDIT: image 2 supplies the target look; image 1 supplies composition and geometry.",
            "Output must look like a real MLS photograph (natural light, real materials)—never cartoon, "
            "illustration, sketch, or heavy HDR outlines.",
            "Keep the same field of view, camera angle, and property layout as image 1 edge to edge.",
            "Change ONLY the user-selected renovation elements; leave every other surface and object as in image 1.",
            f"Renovation type: {type_of_renovation}",
            f"Visual type: {visual_type or 'reference_guided'}",
            f"Desired quality level: {desired_quality_level}",
            f"Visual scope preset: {visual_scope_hint}",
        ]
    else:
        directive_lines = [
            "Full-scene preservation: the output must include the SAME field of view as the source—every pixel "
            "region from the left edge through the right edge. Do not crop, zoom, reframe, or shift the camera. "
            "Keep all doorways, hallway openings, mirrors, windows, ceiling fans, wall art, mobile cabinets/carts, "
            "built-ins, trim, and furniture that appear in the source unless the user explicitly asked to remove them.",
            "Primary directive: execute the explicit user instruction exactly and conservatively.",
            "Hard constraint: change only the specifically requested parts of the image.",
            "Hard constraint: do not add, remove, restage, redesign, or alter any non-requested objects/surfaces.",
            "Preserve original structural material (wood remains wood; concrete remains concrete).",
            f"Renovation type: {type_of_renovation}",
            f"Visual type: {visual_type}",
            f"Desired quality level: {desired_quality_level}",
            f"Visual scope preset: {visual_scope_hint}",
        ]

    if base_instruction:
        _append_authoritative_user_edit_block(directive_lines, base_instruction=base_instruction)

    if reference_image_attached:
        _append_reference_directives(
            directive_lines,
            visual_type=visual_type,
            reference_image_url=reference_image_url,
            reference_image_attached=True,
            element_descriptions=element_descriptions,
            elements=elements,
            type_of_renovation=type_of_renovation,
        )

    issues = [i.strip() for i in (detected_issues or []) if i and i.strip()]
    if issues:
        _append_issue_repair_directives(
            directive_lines,
            issues=issues,
            elements=elements,
            element_descriptions=element_descriptions,
            base_instruction=base_instruction,
        )
    else:
        if elements:
            directive_lines.append(
                "No AI-detected damage issues. Apply user-selected element upgrades only; "
                "do not redesign or restage unselected areas."
            )
        else:
            directive_lines.append(_NO_ISSUES_DETECTED_INSTRUCTION)

    _append_selected_element_directives(
        directive_lines,
        visual_type=visual_type,
        elements=elements,
        element_descriptions=element_descriptions,
        type_of_renovation=type_of_renovation,
        reference_image_attached=reference_image_attached,
    )
    if not reference_image_attached:
        _append_reference_directives(
            directive_lines,
            visual_type=visual_type,
            reference_image_url=reference_image_url,
            reference_image_attached=False,
            element_descriptions=element_descriptions,
            elements=elements,
            type_of_renovation=type_of_renovation,
        )
    _append_user_note_directive(directive_lines, base_instruction=base_instruction)
    if base_instruction:
        directive_lines.append(
            "Final check: honor the USER IMAGE EDIT REQUEST block for every explicit color, finish, "
            "fixture, or decor change the user named."
        )

    return "\n".join(directive_lines)


def _read_prompt_text(path: Path, fallback: str) -> str:
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return fallback


def _load_edit_prompt_text() -> str:
    return _read_prompt_text(
        _EDIT_PROMPT_BASE,
        "You are editing one real-estate photo. Apply only requested visual changes.",
    )


_BROAD_RENOVATION = _read_prompt_text(
    _EDIT_PROMPT_BROAD,
    "MODE: Full renovation / damage repair. Preserve the full frame; fix only visible damage.",
)
_TARGETED_CHANGE = _read_prompt_text(
    _EDIT_PROMPT_TARGETED,
    "MODE: Targeted edit only. Apply only what the user requested; keep the rest of the scene unchanged.",
)
_REFERENCE_STYLE_CHANGE = _read_prompt_text(
    _EDIT_PROMPT_REFERENCE,
    "MODE: Reference-guided transfer from image 2 onto selected elements in image 1. Photorealistic output only.",
)


def _select_constraints_for_instruction(instruction: str, *, reference_attached: bool = False) -> str:
    if reference_attached:
        return _REFERENCE_STYLE_CHANGE
    if (instruction or "").strip():
        return _TARGETED_CHANGE
    return _BROAD_RENOVATION


def _guess_image_filename(media_type: str, *, role: str) -> str:
    mt = (media_type or "image/png").split(";")[0].strip().lower()
    ext = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(mt, "png")
    return f"{role}.{ext}"


async def edit_property_image_from_url(
    *,
    image_url: str,
    instruction: str,
    reference_image_url: str = "",
    preserve_elements: str | None = None,
) -> ImageEditResult:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for image edits.")
    if not image_url.strip():
        raise ValueError("image_url is required.")
    if not instruction.strip():
        raise ValueError("instruction is required.")

    try:
        image_bytes, media_type = await _download_source_image(image_url.strip())
    except Exception as exc:
        logger.warning("Source image download failed for edit (url=%s): %s", image_url, exc)
        raise ValueError(
            "Image URL could not be downloaded for structure-preserving edit. "
            "Check that the URL is public/reachable, or configure IMAGE_DOWNLOAD_REFERER / "
            "IMAGE_DOWNLOAD_PROXY_TEMPLATE for blocked CDNs."
        ) from exc

    image_files: list[tuple[str, bytes, str]] = [
        (_guess_image_filename(media_type, role="property_image"), image_bytes, media_type),
    ]
    ref_url = (reference_image_url or "").strip()
    if ref_url:
        try:
            ref_bytes, ref_media = await _download_source_image(ref_url)
            image_files.append(
                (_guess_image_filename(ref_media, role="reference_image"), ref_bytes, ref_media)
            )
        except Exception as exc:
            logger.warning("Reference image download failed for edit (url=%s): %s", ref_url, exc)
            raise ValueError(
                "Reference image URL could not be downloaded. "
                "Check that the URL is public/reachable, or configure IMAGE_DOWNLOAD_REFERER / "
                "IMAGE_DOWNLOAD_PROXY_TEMPLATE for blocked CDNs."
            ) from exc

    reference_attached = len(image_files) > 1
    prompt_template = _load_edit_prompt_text()
    constraints = (
        preserve_elements.strip()
        if preserve_elements is not None
        else _select_constraints_for_instruction(instruction.strip(), reference_attached=reference_attached)
    )
    if reference_attached:
        prompt = (
            f"{prompt_template}\n\n"
            "REFERENCE EDIT: Two images are attached—property first, reference second. "
            "Follow the Constraints block for how to use image 2.\n\n"
            f"User request:\n{instruction.strip()}\n\n"
            f"Constraints:\n{constraints}"
        )
    else:
        prompt = f"{prompt_template}\n\nUser request: {instruction.strip()}\n\nConstraints:\n{constraints}"

    timeout_sec = float(settings.OPENAI_IMAGE_EDIT_TIMEOUT_SECONDS)
    max_retries = int(settings.OPENAI_IMAGE_EDIT_MAX_RETRIES)

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=timeout_sec,
        max_retries=0,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            model = settings.default_openai_image_edit_model
            edit_image: tuple[str, bytes, str] | list[tuple[str, bytes, str]] = (
                image_files if len(image_files) > 1 else image_files[0]
            )
            edit_kwargs: dict = {
                "model": model,
                "image": edit_image,
                "prompt": prompt,
                "size": _select_gpt_image_edit_size(image_bytes),
                "quality": "high",
            }
            if _image_edit_supports_input_fidelity(model):
                # High fidelity locks pixels and blocks visible material swaps; use low when a reference is attached.
                edit_kwargs["input_fidelity"] = "low" if reference_attached else "high"
            response = await client.images.edit(**edit_kwargs)

            data = getattr(response, "data", None) or []
            if not data or not getattr(data[0], "b64_json", None):
                raise ValueError("Image edit failed: no image returned.")
            first = data[0]
            revised_prompt = getattr(first, "revised_prompt", "") or prompt
            return ImageEditResult(
                revised_prompt=revised_prompt,
                image_base64=first.b64_json,
                media_type="image/png",
            )
        except Exception as exc:
            last_exc = exc
            is_retryable = _is_retryable_openai_exception(exc)
            if attempt < max_retries and is_retryable:
                await _wait_for_retry_backoff(attempt)
                continue
            logger.warning(
                "OpenAI images.edit failed after %s attempt(s) (timeout=%ss, model=%s): %s: %s",
                attempt + 1,
                timeout_sec,
                settings.default_openai_image_edit_model,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise ValueError(
                "Image edit request failed. Retry later. "
                "If this repeats, inspect server logs for the upstream OpenAI error."
            ) from exc

    raise ValueError("Image edit request failed.") from last_exc


async def _download_source_image(image_url: str) -> tuple[bytes, str]:
    cached = redis_cache.get_cached_image_download(
        image_url,
        ttl_seconds=IMAGE_CACHE_TTL_SECONDS,
        cache_dir=IMAGE_CACHE_DIR,
    )
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(image_url, headers=_build_image_download_headers(image_url))
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        proxy_urls = _build_proxy_image_urls(image_url)
        if proxy_urls and status in (401, 403):
            try:
                last_proxy_exc: Exception | None = None
                for proxy_url in proxy_urls:
                    try:
                        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                            response = await client.get(proxy_url, headers={"User-Agent": _BROWSER_USER_AGENT})
                            response.raise_for_status()
                        break
                    except Exception as proxy_exc:
                        last_proxy_exc = proxy_exc
                        continue
                else:
                    raise last_proxy_exc or ValueError("Unknown proxy download failure")
            except Exception as proxy_exc:
                raise ValueError(
                    f"Image URL could not be downloaded (HTTP {status}), and proxy fallback failed: {proxy_exc}"
                ) from proxy_exc
        else:
            raise ValueError(
                f"Image URL could not be downloaded (HTTP {status}). "
                "Some CDNs block non-browser clients; configure IMAGE_DOWNLOAD_REFERER or IMAGE_DOWNLOAD_PROXY_TEMPLATE."
            ) from exc
    except httpx.HTTPError as exc:
        raise ValueError(
            "Image URL could not be downloaded. "
            "Check that the URL is public and reachable."
        ) from exc

    media_type = response.headers.get("content-type", "image/png").split(";")[0].strip() or "image/png"
    if not media_type.startswith("image/"):
        raise ValueError("URL did not return an image.")
    if len(response.content) > 20 * 1024 * 1024:
        raise ValueError("Image is too large. Max supported size is 20MB.")
    redis_cache.set_cached_image_download(
        image_url,
        content=response.content,
        media_type=media_type,
        ttl_seconds=IMAGE_CACHE_TTL_SECONDS,
        cache_dir=IMAGE_CACHE_DIR,
    )
    return response.content, media_type


async def image_url_as_openai_vision_payload(image_url: str) -> str:
    """
    Prefer a data URL for OpenAI vision so their servers do not fetch the URL.

    Presigned object-storage URLs often time out from OpenAI (400 invalid_request_error).
    """
    url = (image_url or "").strip()
    if not url:
        return ""
    try:
        image_bytes, media_type = await _download_source_image(url)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{media_type};base64,{b64}"
    except Exception as exc:
        logger.warning(
            "OpenAI vision: server-side download failed (%s): %s",
            url[:120] + ("…" if len(url) > 120 else ""),
            exc,
        )
        return url
