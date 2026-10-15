"""Optional cloud vision and artwork generation for ThreadOps."""
from __future__ import annotations

from io import BytesIO
import os
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from google import genai

VISION_MODEL = "gemini-3.8-flash"
IMAGE_MODEL = "gemini-nano-banana-2.1"


def get_client(api_key: str | None = None) -> genai.Client:
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise ValueError("Enter a Gemini API key or set GEMINI_API_KEY.")
    from google import genai

    return genai.Client(api_key=key, vertexai=False)


def analyze_and_prompt(image: Image.Image, style: str,
                       api_key: str | None = None) -> str:
    if not isinstance(image, Image.Image):
        raise ValueError("Upload a customer photo first.")
    client = get_client(api_key)
    from google.genai import types

    # Flatten transparency onto white before encoding as JPEG.
    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG")
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            "Act as an apparel digitizer art director. Describe the supplied photo "
            f"as a concise vector/embroidery artwork prompt for this style: {style}. "
            "Capture the subject's breed (if applicable), colors, distinctive markings, "
            "and posture. Use at most 20 words so it fits the local image model. Return only the descriptive prompt.",
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
        ],
    )
    prompt = (response.text or "").strip()
    if not prompt:
        raise ValueError("Gemini returned no description. Try another photo.")
    return " ".join(prompt.split()[:20])


def generate_with_gemini(prompt: str, api_key: str | None = None,
                         image: Image.Image | None = None) -> Image.Image:
    """Generate artwork through the Developer API, optionally using a photo reference."""
    client = get_client(api_key)
    from google.genai import types

    contents = [prompt]
    if image is not None:
        contents.append(image)
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="1:1"),
        ),
    )
    for part in response.parts or []:
        if part.thought:
            continue
        data = part.inline_data
        if data and data.mime_type and data.mime_type.startswith("image/") and data.data:
            with Image.open(BytesIO(data.data)) as artwork:
                return artwork.copy()
    raise ValueError("Gemini returned no artwork. The request may have been blocked; try a different prompt.")


def generate_with_imagen(prompt: str, api_key: str | None = None) -> Image.Image:
    """Compatibility wrapper for callers using the retired Imagen service name."""
    return generate_with_gemini(prompt, api_key)
