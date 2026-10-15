"""Real local quality comparison with a fixed seed; no downloads or cloud calls."""
import time
from PIL import Image
import pipeline


if __name__ == "__main__":
    pipeline.warmup()
    profile = pipeline.MODEL_ID.replace("/", "--").replace("\\", "--").replace(":", "")
    output = pipeline.ROOT / "outputs" / "portrait-comparison" / profile
    output.mkdir(parents=True, exist_ok=True)
    source = Image.open(pipeline.ROOT / "samples" / "collie_photo_input.jpg")
    pipeline.preprocess(source).save(output / "prepared.png")
    for strength in (0.50, 0.60, 0.70):
        started = time.perf_counter()
        edges, art = pipeline.generate(
            source, next(iter(pipeline.STYLE_PROMPTS)),
            custom_prompt="tri-color collie, tan cheeks, black ears, white chest",
            control_strength=0.95, denoise_strength=strength, steps=30, seed=42,
        )
        edges.save(output / "contours.png")
        art.save(output / f"portrait-{strength:.2f}.png")
        print(f"strength={strength}: {time.perf_counter() - started:.1f}s", flush=True)
