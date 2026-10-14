"""Local SD 1.5 image-to-image + Canny ControlNet for an 8 GB GPU."""
from __future__ import annotations

import logging
import math
import os
import secrets
import threading
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".models" / "huggingface"))
os.environ.setdefault("U2NET_HOME", str(ROOT / ".models" / "rembg"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

LOGGER = logging.getLogger(__name__)
MODEL_ID = os.getenv("THREADOPS_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_canny"
OFFLINE = os.getenv("THREADOPS_OFFLINE", "0") == "1"
_PIPELINE = None
_REMBG_SESSION = None
_REMBG_ERROR = "Background session has not been warmed up."
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()

STYLE_PROMPTS = {
    "🐶 Flat Vector Pet Portrait": {
        "prompt": "clean flat vector art, vector illustration of pet portrait, bold smooth outlines, flat cel shading, vibrant solid color blocks, minimalist graphic art, sticker design, sharp details, pure white background, masterpiece",
        "negative": "photorealistic, 3d render, photograph, noisy, blurry, photographic fur texture, messy gradients, realistic skin, complex shading, dark background",
        "default_denoise": 0.70, "default_control": 0.85, "default_guidance": 7.5,
    },
    "👫 Faceless Minimalist Couple Art": {
        "prompt": "faceless minimalist vector illustration of a couple, bold clean black outlines, faceless aesthetic without eyes or nose, flat pastel and solid clothing colors, detailed hair and clothing contours, modern portrait art, pure white background",
        "negative": "eyes, nose, mouth, realistic face, photorealistic, 3d render, photo, messy lines, realistic skin texture, noisy, textured background",
        "default_denoise": 0.72, "default_control": 0.90, "default_guidance": 7.5,
    },
    "🧵 Embroidery Thread Simulation": {
        "prompt": "detailed embroidered patch, machine embroidery stitch simulation, realistic embroidery thread texture, raised thread relief, satin stitch, chenille patch, textile embroidery art, crisp patch border, pure white background",
        "negative": "flat paper, smooth drawing, photo, blurry, realistic dog fur, gradient shading, noisy",
        "default_denoise": 0.75, "default_control": 0.80, "default_guidance": 7.5,
    },
}


def device_description() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} · CUDA {torch.version.cuda} · FP16 / CPU offload"
    return "CUDA unavailable — install the CUDA 12.4 PyTorch wheels and check your NVIDIA driver"


def white_canvas(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def create_rembg_session(local_files_only: bool = True):
    """Only the preloader may download; startup and requests remain offline."""
    cache = Path(os.environ["U2NET_HOME"])
    if local_files_only and not any(path.is_file() for path in (
        cache / "models" / "u2netp" / "u2netp.onnx", cache / "u2netp.onnx",
    )):
        raise FileNotFoundError("u2netp.onnx is missing. Run python preload_models.py first.")
    from rembg import new_session
    return new_session(model_name="u2netp", providers=["CPUExecutionProvider"])


def remove_background(image: Image.Image, notices: list[str] | None = None) -> Image.Image:
    """Use the warmed CPU-only session; never download inside a user request."""
    try:
        if _REMBG_SESSION is None:
            raise RuntimeError(_REMBG_ERROR)
        from rembg import remove
        return white_canvas(remove(image, session=_REMBG_SESSION))
    except Exception as exc:
        LOGGER.warning("Background isolation failed: %s", exc)
        if notices is not None:
            notices.append("Background removal unavailable; original photo retained on a white canvas.")
        return white_canvas(image)


def preprocess(image: Image.Image, isolate_background: bool = True,
               notices: list[str] | None = None) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    image = remove_background(image, notices) if isolate_background else white_canvas(image)
    # Padding, rather than stretching, preserves the subject's anatomy.
    return ImageOps.pad(image, (512, 512), method=Image.Resampling.LANCZOS, color="white")


def extract_canny_edges(image: Image.Image, low_threshold: int = 100,
                        high_threshold: int = 200) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return Image.fromarray(cv2.Canny(blurred, low_threshold, high_threshold)).convert("RGB")


def _bold_font(size: int):
    for path in (ROOT / "assets" / "varsity.ttf",
                 Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "ariblk.ttf",
                 Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf",
                 Path("DejaVuSans-Bold.ttf")):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def add_text_overlay(image: Image.Image, text: str, font_size: int = 48,
                     position: str = "Bottom") -> Image.Image:
    """Fit bold caps lettering and a contrasting stroke into a white name band."""
    clean = " ".join(text.split()).upper()
    if not clean:
        return image
    if len(clean) > 80:
        raise ValueError("Keep the name banner to 80 characters or fewer.")
    if position not in ("Top", "Bottom"):
        raise ValueError("Text position must be Top or Bottom.")
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    size = font_size
    while True:
        font = _bold_font(size)
        stroke = max(1, size // 14)
        box = draw.textbbox((0, 0), clean, font=font, stroke_width=stroke)
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= result.width - 32 or size <= 8:
            break
        size -= 1
    if width > result.width - 32:
        raise ValueError("Name is too wide; shorten the banner text.")
    band = max(64, height + 28)
    y0 = 0 if position == "Top" else result.height - band
    draw.rectangle((0, y0, result.width, y0 + band), fill="white")
    x = (result.width - width) / 2 - box[0]
    y = y0 + (band - height) / 2 - box[1]
    draw.text((x, y), clean, font=font, fill="#182f2a",
              stroke_width=stroke, stroke_fill="#d7b77a")
    return result


def load_diffusion_models():
    """Read FP16 safetensors strictly from cache. Used by preloader and startup."""
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    options = dict(torch_dtype=torch.float16, variant="fp16",
                   use_safetensors=True, local_files_only=True)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, **options)
    return StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, safety_checker=None,
        requires_safety_checker=False, **options,
    )


def warmup():
    """Explicit startup initialization. No network access or request-time loading."""
    global _PIPELINE
    global _REMBG_SESSION, _REMBG_ERROR
    with _MODEL_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA GPU not detected. Install requirements.txt into .venv and check nvidia-smi.")
        try:
            print("[Startup] Loading cached FP16 SD 1.5 + Canny ControlNet...", flush=True)
            pipe = load_diffusion_models()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load cached FP16 models ({MODEL_ID}). Run python preload_models.py before launching. "
                "THREADOPS_MODEL_ID may point to a local Diffusers folder with FP16 variant weights. "
                f"Details: {exc}"
            ) from exc
        try:
            print("[Startup] Initializing u2netp with CPUExecutionProvider...", flush=True)
            _REMBG_SESSION = create_rembg_session(local_files_only=True)
            _REMBG_ERROR = ""
        except Exception as exc:
            _REMBG_ERROR = str(exc)
            LOGGER.warning("Background isolation disabled: %s", exc)
        _PIPELINE = pipe
        print("[Ready] " + device_description(), flush=True)
        return pipe


def get_pipeline():
    """Accessor only: requests must never initialize or download models."""
    if _PIPELINE is None:
        raise RuntimeError("Pipeline is not ready. Run preload_models.py, then call warmup() before serving requests.")
    return _PIPELINE


def generate(input_image: Image.Image, style_choice: str, custom_prompt: str = "",
             text_overlay: str = "", control_strength: float | None = None,
             denoise_strength: float | None = None, steps: int = 25,
             guidance_scale: float = 7.5, isolate_background: bool = True,
             text_position: str = "Bottom", seed: int = -1,
             progress: Callable[[float, str], None] | None = None,
             notices: list[str] | None = None,
             step_callback: Callable[[int, int], None] | None = None):
    """Return (RGB Canny map, artwork). All inference stays on this machine."""
    if not isinstance(input_image, Image.Image):
        raise ValueError("Upload a customer photo first.")
    if style_choice not in STYLE_PROMPTS:
        raise ValueError("Choose one of the three supported art styles.")
    preset = STYLE_PROMPTS[style_choice]
    control = preset["default_control"] if control_strength is None else float(control_strength)
    strength = preset["default_denoise"] if denoise_strength is None else float(denoise_strength)
    if not math.isfinite(control) or not 0.1 <= control <= 1.5:
        raise ValueError("Contour strictness must be between 0.1 and 1.5.")
    if not math.isfinite(strength) or not 0.2 <= strength <= 1.0:
        raise ValueError("Style strength must be between 0.2 and 1.0.")
    if int(steps) != steps or not 20 <= steps <= 30:
        raise ValueError("Inference steps must be an integer from 20 to 30.")
    if not math.isfinite(float(guidance_scale)) or not 1 <= guidance_scale <= 15:
        raise ValueError("Guidance must be between 1 and 15.")
    if int(seed) != seed or not -1 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be -1 (random) or an integer from 0 to 4294967295.")
    if len(" ".join(text_overlay.split())) > 80:
        raise ValueError("Keep the name banner to 80 characters or fewer.")
    if text_position not in ("Top", "Bottom"):
        raise ValueError("Text position must be Top or Bottom.")
    report = progress or (lambda value, description: None)
    used_seed = secrets.randbelow(2**32) if seed == -1 else int(seed)
    if notices is not None:
        notices.append(f"Seed: {used_seed}.")
    with _GENERATION_LOCK:
        try:
            pipe = get_pipeline()
            report(0.0, "Starting local generation")
            report(0.02, "Preparing photo / isolating background")
            source = preprocess(input_image, isolate_background, notices)
            edges = extract_canny_edges(source)
            prompt = preset["prompt"]
            if custom_prompt.strip():
                prompt += ", " + custom_prompt.strip()
            report(0.18, "Generating artwork on CUDA")

            def on_step(pipeline, step_index, timestep, callback_kwargs):
                total = max(1, int(pipeline.num_timesteps))
                report(min(0.95, 0.18 + 0.76 * (step_index + 1) / total),
                       f"Diffusion step {step_index + 1}/{total}")
                if step_callback is not None:
                    step_callback(step_index + 1, total)
                return callback_kwargs

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt, negative_prompt=preset["negative"],
                    image=source, control_image=edges, strength=strength,
                    num_inference_steps=int(steps), guidance_scale=float(guidance_scale),
                    controlnet_conditioning_scale=control,
                    generator=torch.Generator(device="cpu").manual_seed(used_seed),
                    callback_on_step_end=on_step,
                ).images[0]
            report(0.98, "Adding name banner")
            final = add_text_overlay(result, text_overlay, position=text_position)
            report(1.0, "Artwork complete")
            return edges, final
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU memory exhausted. Close other GPU apps and retry with CPU offload enabled.") from exc
        finally:
            if _PIPELINE is not None:
                _PIPELINE.maybe_free_model_hooks()
