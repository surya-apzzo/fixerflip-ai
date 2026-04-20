from dataclasses import dataclass


@dataclass(frozen=True)
class ImageEditResult:
    revised_prompt: str
    image_base64: str
    media_type: str = "image/png"
