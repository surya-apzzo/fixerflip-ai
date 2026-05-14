"""Image edit engine: OpenAI -> Image."""
from __future__ import annotations

import asyncio
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

_OPENAI_IMAGE_EDIT_TIMEOUT_SECONDS = 60.0
_OPENAI_IMAGE_EDIT_MAX_RETRIES = 1
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
    "paint": "wall and ceiling paint finish",
    "lighting": "light fixtures and lighting look",
    "furniture": "furniture style and condition",
    "roof": "roof shingles and roofline finish",
    "cabinet": "kitchen or bathroom cabinetry fronts",
    "window": "window frames and glass panes",
    "stair": "staircase steps, railing, and finish",
    "door": "interior and exterior doors",
}

_ELEMENT_ACTION_HINTS = {
    "flooring": (
        "Flooring directive: visibly replace the existing floor finish in all clearly visible walkable floor areas "
        "with a new tile look that is materially different from the source floor. Keep perspective and room geometry "
        "identical, and do not alter cabinets, counters, walls, ceiling, appliances, furniture, or decor."
    ),
    "paint": (
        "Paint directive: visibly update only the requested wall/ceiling paint surfaces while keeping all other finishes unchanged."
    ),
    "lighting": (
        "Lighting directive: update only visible light fixtures and lighting tone/intensity while preserving all cabinetry, "
        "walls, flooring, furniture, and room geometry."
    ),
    "furniture": (
        "Furniture directive: update only furniture look/style/finish of existing furniture in place. "
        "Do not add extra furniture pieces and do not alter architecture, cabinetry, walls, or flooring."
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
) -> None:
    if visual_type != "select_elements_to_renovate":
        return
    if element_descriptions:
        directive_lines.append(
            "Selected elements to renovate: " + ", ".join(element_descriptions) + "."
        )
        directive_lines.append("Apply visible changes only to selected elements.")
        directive_lines.append(
            "Do NOT modify unselected elements (especially wall color, ceiling, flooring, cabinets, counters, appliances) unless explicitly selected."
        )
        directive_lines.append(
            "At least one visible change must be made for each selected element while preserving the original room layout."
        )
        for element in elements:
            action_hint = _ELEMENT_ACTION_HINTS.get(element)
            if action_hint:
                directive_lines.append(action_hint)
        return
    directive_lines.append(
        "No specific elements were selected. Use conservative damage-repair mode only."
    )
    directive_lines.append(
        "Keep all intact property features unchanged (especially cabinets, counters, flooring, walls, ceilings, fixtures, appliances, furniture, and decor)."
    )


def _append_reference_directives(
    directive_lines: list[str],
    *,
    visual_type: str,
    reference_image_url: str,
) -> None:
    if visual_type == "upload_my_own_reference_photo" and reference_image_url:
        directive_lines.append(f"Reference image URL: {reference_image_url}")
        directive_lines.append(
            "Use the reference image as style guidance for color/material/finish while keeping source structure unchanged."
        )
    elif reference_image_url:
        directive_lines.append(f"Reference image URL: {reference_image_url}")


def _append_user_note_directive(directive_lines: list[str], *, base_instruction: str) -> None:
    if base_instruction:
        directive_lines.append("User instruction (highest priority): " + base_instruction)
    else:
        directive_lines.append("User note: " + _DEFAULT_IMAGE_EDIT_INSTRUCTION)


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
    element_descriptions = [_ELEMENT_HINTS.get(e, e) for e in elements]

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
        directive_lines.append(_NO_ISSUES_DETECTED_INSTRUCTION)

    _append_selected_element_directives(
        directive_lines,
        visual_type=visual_type,
        elements=elements,
        element_descriptions=element_descriptions,
    )
    _append_reference_directives(
        directive_lines,
        visual_type=visual_type,
        reference_image_url=reference_image_url,
    )
    _append_user_note_directive(directive_lines, base_instruction=base_instruction)

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

def _select_constraints_for_instruction(instruction: str) -> str:
    # Any non-empty instruction should default to strict targeted-edit mode.
    if (instruction or "").strip():
        return _TARGETED_CHANGE
    return _BROAD_RENOVATION


async def edit_property_image_from_url(
    *,
    image_url: str,
    instruction: str,
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

    prompt_template = _load_edit_prompt_text()
    constraints = (
        preserve_elements.strip()
        if preserve_elements is not None
        else _select_constraints_for_instruction(instruction.strip())
    )
    prompt = f"{prompt_template}\n\nUser request: {instruction.strip()}\n\nConstraints:\n{constraints}"

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=_OPENAI_IMAGE_EDIT_TIMEOUT_SECONDS,
        max_retries=0,
    )

    last_exc: Exception | None = None
    for attempt in range(_OPENAI_IMAGE_EDIT_MAX_RETRIES + 1):
        try:
            model = settings.default_openai_image_edit_model
            edit_kwargs: dict = {
                "model": model,
                "image": ("property_image.png", image_bytes, media_type),
                "prompt": prompt,
                "size": _select_gpt_image_edit_size(image_bytes),
                "quality": "high",
            }
            if _image_edit_supports_input_fidelity(model):
                edit_kwargs["input_fidelity"] = "high"
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
            if attempt < _OPENAI_IMAGE_EDIT_MAX_RETRIES and is_retryable:
                await _wait_for_retry_backoff(attempt)
                continue
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
