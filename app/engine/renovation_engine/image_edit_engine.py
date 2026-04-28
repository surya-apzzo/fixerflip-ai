"""Image edit engine: OpenAI -> Image."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from app.core import redis_cache
from app.core.config import settings
from app.schemas import ImageEditResult

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
    "Apply light cosmetic cleanup only: remove minor surface debris, "
    "freshen paint where visibly worn, and clean surfaces. "
    "Do not redesign, restage, or make structural changes. "
    "Preserve all existing finishes, furniture, and room layout exactly."
)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


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


_ELEMENT_HINTS = {
    "flooring": "floors and floor finish",
    "paint": "wall and ceiling paint finish",
    "lighting": "light fixtures and lighting look",
    "furniture": "furniture style and condition",
    "kitchen": "kitchen cabinets, counters, and fixtures",
    "bathroom": "bathroom vanity, tile, and fixtures",
    "windows": "window frames and glass appearance",
    "doors": "interior and exterior door appearance",
    "plumbing": "visible sinks, faucets, and plumbing fixtures",
}


def build_instruction_for_edit(
    *,
    user_inputs: str,
    type_of_renovation: str,
    visual_type: str,
    desired_quality_level: str,
    resolved_target_style: str,
    reference_image_url: str = "",
    renovation_elements: list[str] | None = None,
    detected_issues: list[str] | None = None,
) -> str:
    base_instruction = (user_inputs or "").strip()
    elements = [e.strip().lower() for e in (renovation_elements or []) if e and e.strip()]
    element_descriptions = [_ELEMENT_HINTS.get(e, e) for e in elements]

    directive_lines = [
        "Primary directive: follow renovation type, visual type, and selected elements first.",
        "If any user text conflicts with selected elements, selected elements take priority.",
        "Preserve original structural material (wood remains wood; concrete remains concrete).",
        f"Renovation type: {type_of_renovation}",
        f"Visual type: {visual_type}",
        f"Desired quality level: {desired_quality_level}",
        f"Target renovation style: {resolved_target_style}",
    ]

    issues = [i.strip() for i in (detected_issues or []) if i and i.strip()]
    if issues:
        directive_lines.append(
            "AI-detected issues/damages to repair: " + ", ".join(issues) + "."
        )
        directive_lines.append(
            "Focus repairs on the above detected issues. Fix each one while preserving "
            "all undamaged areas of the image."
        )
    else:
        directive_lines.append(_NO_ISSUES_DETECTED_INSTRUCTION)

    if visual_type == "select_elements_to_renovate":
        if element_descriptions:
            directive_lines.append(
                "Selected elements to renovate: " + ", ".join(element_descriptions) + "."
            )
            directive_lines.append("Apply visible changes only to selected elements.")
            directive_lines.append(
                "Do NOT modify unselected elements (especially wall color, ceiling, flooring, cabinets, counters, appliances) unless explicitly selected."
            )
        else:
            directive_lines.append(
                "No specific elements were selected; apply a balanced full-scene renovation while preserving layout."
            )

    if visual_type == "upload_my_own_reference_photo" and reference_image_url:
        directive_lines.append(f"Reference image URL: {reference_image_url}")
        directive_lines.append(
            "Use the reference image as style guidance for color/material/finish while keeping source structure unchanged."
        )
    elif reference_image_url:
        directive_lines.append(f"Reference image URL: {reference_image_url}")

    if base_instruction:
        directive_lines.append(
            "User note (apply only if consistent with selected elements): " + base_instruction
        )
    else:
        directive_lines.append("User note: " + _DEFAULT_IMAGE_EDIT_INSTRUCTION)

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

_TARGETED_SCOPE_RE = tuple(
    re.compile(p)
    for p in (
        r"\b(?:change|paint|replace|update)\s+only\b",
        r"\bonly\s+(?:the|change|paint|replace|update)\b",
        r"\bsolely\s+the\b",
        r"\bjust\s+(the\s+)?(?:wall|walls|floor|flooring|ceiling|trim|baseboard|backsplash|cabinet|cabinets|counter|countertop|window|windows|door|doors|paint|color)\b",
        r"\bjust\s+(?:change|paint|replace|update)\b",
        r"\b(?:don'?t|do not)\s+(?:touch|change|alter|fix|repair)\b",
        r"\bleave\s+(?:everything\s+else|the\s+rest|untouched)\b",
        r"\breplace\s+only\b",
        r"\b(?:change|paint|repaint)\s+(?:the\s+)?(?:wall|walls|ceiling|trim)\b",
        r"\bwall\s+(?:color|colour)\b",
        r"\b(?:ceiling|trim)\s+(?:color|colour)\b",
    )
)


def _is_targeted_edit_instruction(instruction: str) -> bool:
    """True if the user narrowed scope to a specific edit; otherwise use broad damage-repair mode."""
    t = (instruction or "").strip().lower()
    if not t:
        return False
    return any(rx.search(t) for rx in _TARGETED_SCOPE_RE)


def _select_constraints_for_instruction(instruction: str) -> str:
    if _is_targeted_edit_instruction(instruction):
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

    image_bytes, media_type = await _download_source_image(image_url.strip())
    prompt_template = _load_edit_prompt_text()
    constraints = (
        preserve_elements.strip()
        if preserve_elements is not None
        else _select_constraints_for_instruction(instruction.strip())
    )
    prompt = f"{prompt_template}\n\nUser request: {instruction.strip()}\n\nConstraints:\n{constraints}"

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.images.edit(
        model=settings.default_openai_image_edit_model,
        image=("property_image.png", image_bytes, media_type),
        prompt=prompt,
        size="auto",
        input_fidelity="high",
        quality="high",
    )

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
        raise ValueError(
            f"Image URL could not be downloaded (HTTP {status}). "
            "Some CDNs block non-browser clients; this server sends a browser-like User-Agent. "
            "If it still fails, try IMAGE_DOWNLOAD_REFERER or a signed/proxy URL from your MLS provider."
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
