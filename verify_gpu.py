"""Optional real GPU smoke test. Downloads models on first use; saves PNGs."""
import time
from PIL import Image
import pipeline


if __name__ == "__main__":
    started = time.perf_counter()
    pipeline.warmup()
    notices = []
    edges, artwork = pipeline.generate(
        Image.open(pipeline.ROOT / "samples" / "collie_photo_input.jpg"),
        next(iter(pipeline.STYLE_PROMPTS)), text_overlay="OREO", steps=20, seed=42,
        notices=notices,
        progress=lambda value, description: print(f"{value:.0%}: {description}", flush=True),
    )
    output = pipeline.ROOT / "outputs"
    output.mkdir(exist_ok=True)
    artwork.save(output / "gpu-smoke-artwork.png")
    edges.save(output / "gpu-smoke-contours.png")
    print(f"Completed in {time.perf_counter() - started:.1f}s; {pipeline.device_description()}")
    print(" ".join(notices))
