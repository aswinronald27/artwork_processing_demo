"""Visual experiment; saves candidates without changing application defaults."""
from PIL import Image
import pipeline
from diffusers import DPMSolverMultistepScheduler


if __name__ == "__main__":
    pipe = pipeline.warmup()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )
    preset = pipeline.STYLE_PROMPTS[next(iter(pipeline.STYLE_PROMPTS))]
    preset["prompt"] = "vector art, flat colors, 2D illustration, a single collie dog head portrait, symmetrical face, angular fur tufts made of solid color shapes, clean sharp edges, limited color palette, isolated on pure white background"
    preset["negative"] += ", realistic fur, realistic shading, soft shading, highlights, painterly, oil painting, airbrush, gradients, 3D lighting, body, paws, ornate, background shapes"
    image = Image.open(pipeline.ROOT / "samples" / "collie_photo_input.jpg")
    output = pipeline.ROOT / "outputs" / "vector-benchmark"
    output.mkdir(parents=True, exist_ok=True)
    pipeline.preprocess(image).save(output / "prepared.png")
    for strength, control in [(0.8, 0.65), (0.95, 0.65), (0.85, 0.85)]:
        edges, artwork = pipeline.generate(
            image, next(iter(pipeline.STYLE_PROMPTS)), steps=30, seed=42,
            denoise_strength=strength, control_strength=control, guidance_scale=9,
        )
        name = f"{strength}-{control}"
        artwork.save(output / f"{name}.png")
        edges.save(output / "contours.png")
        print(f"Saved {name}", flush=True)
