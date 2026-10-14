"""Gradio dashboard for ThreadOps; only binds to localhost."""
from __future__ import annotations

import os
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import pipeline
import gradio as gr

STYLES = list(pipeline.STYLE_PROMPTS)
CSS = """
.gradio-container {max-width: 1360px !important; margin: auto;}
#studio-header {padding: 22px 0 18px; border-bottom: 1px solid #cdd6cf; margin-bottom: 16px;}
#studio-header h1 {font-size: clamp(24px, 3vw, 36px); letter-spacing: -1px;}
#studio-header p {max-width: 800px;}
#studio-status {padding: 12px 16px; border: 1px solid #cdd6cf; border-radius: 8px; min-height: 70px;}
footer {display: none !important;}
"""


def on_style_change(style_name):
    preset = pipeline.STYLE_PROMPTS[style_name]
    return preset["default_control"], preset["default_denoise"]


def run_generation(image, style, custom_prompt, text_overlay, control_scale,
                   denoise_strength, steps, guidance_scale, remove_bg,
                   text_position="Bottom", seed=-1, progress=gr.Progress(track_tqdm=True)):
    """Stream status; Gradio's progress display supplies a live elapsed timer."""
    if image is None:
        yield None, None, "**Photo needed.** Upload an image or select a sample below."
        return
    started = time.perf_counter()
    yield None, None, "**Generating…** " + pipeline.device_description() + "  \nFollow the live diffusion progress and elapsed timer above."
    notices = []
    try:
        contours, artwork = pipeline.generate(
            input_image=image, style_choice=style, custom_prompt=custom_prompt,
            text_overlay=text_overlay, control_strength=control_scale,
            denoise_strength=denoise_strength, steps=steps,
            guidance_scale=guidance_scale, isolate_background=remove_bg,
            text_position=text_position, seed=seed, notices=notices,
            progress=lambda value, description: progress(value, desc=description),
        )
        elapsed = time.perf_counter() - started
        status = f"**Completed in {elapsed:.1f}s** · {artwork.width} × {artwork.height} PNG  \n{pipeline.device_description()}  \n" + " ".join(notices)
        yield artwork, contours, status
    except Exception as exc:
        gr.Warning(str(exc))
        yield None, None, f"**Generation failed after {time.perf_counter() - started:.1f}s.** See the error notification and terminal for details."


def example_rows():
    samples = pipeline.ROOT / "samples"
    rows = [
        ["collie_photo_input.jpg", STYLES[0], "tri-color collie, white chest, dark eye patches", "", 0.85, 0.70],
        ["lab_photo_input.jpg", STYLES[0], "chocolate labrador, white muzzle", "OREO", 0.85, 0.70],
        ["couple_boat_input.jpg", STYLES[1], "couple on a boat, preserve hair and clothing silhouettes", "MUMMY & DADDY", 0.90, 0.72],
        ["collie_photo_input.jpg", STYLES[2], "tri-color collie patch, satin stitches", "OREO", 0.80, 0.75],
    ]
    return [[str(samples / row[0]), *row[1:]] for row in rows if (samples / row[0]).is_file()]


def build_ui():
    with gr.Blocks(title="ThreadOps AI Art Studio (Local GPU)", analytics_enabled=False) as demo:
        gr.Markdown(
            "# ThreadOps AI Art Studio (Local GPU)\n"
            "Customer photo → apparel artwork. Three art treatments, precise contour guidance, and a personal name banner. "
            "**Generated on your machine.**", elem_id="studio-header",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"],
                                 label="Customer photo", height=300, format="png", image_mode="RGBA")
                style = gr.Dropdown(STYLES, value=STYLES[0], label="Art style")
                custom = gr.Textbox(label="Custom prompt additions", placeholder="e.g. black ears, white chest, red collar", lines=2)
                text = gr.Textbox(label="Name banner (optional)", placeholder="OREO / MUMMY & DADDY", max_length=80)
                with gr.Accordion("Advanced settings", open=False):
                    remove_bg = gr.Checkbox(value=True, label="Isolate subject on white (rembg)")
                    control = gr.Slider(0.1, 1.5, value=0.85, step=0.01, label="Contour strictness / likeness")
                    strength = gr.Slider(0.2, 1.0, value=0.70, step=0.01, label="Style transfer strength",
                                         info="Higher values redraw more of the original photo.")
                    steps = gr.Slider(20, 30, value=25, step=1, label="Inference steps")
                    guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Prompt guidance")
                    position = gr.Radio(["Top", "Bottom"], value="Bottom", label="Name placement")
                    seed = gr.Number(value=-1, precision=0, minimum=-1, maximum=4294967295, label="Seed (-1 = random)")
                generate = gr.Button("Generate Digitizer Art", variant="primary", size="lg")
            with gr.Column(scale=6, min_width=320):
                status = gr.Markdown("**Ready** · " + pipeline.device_description(), elem_id="studio-status")
                with gr.Row():
                    artwork = gr.Image(label="Generated Artwork", type="pil", format="png", interactive=False, height=460)
                    contours = gr.Image(label="Detected Contours (ControlNet Canny Map)", type="pil", format="png", interactive=False, height=460)
                gr.Markdown("Download the PNG from its preview. Artwork is a **raster reference for your digitizer**; embroidery stitch files and SVG paths require a separate digitizing step.")
        examples = example_rows()
        if examples:
            gr.Examples(examples=examples, inputs=[image, style, custom, text, control, strength],
                        label="Sample photos · click to load, then Generate", cache_examples=False)
        gr.Markdown("Models are preloaded before startup. Photos and generation stay local; no paid API is used.")
        style.change(on_style_change, inputs=style, outputs=[control, strength], queue=False)
        generate.click(
            run_generation,
            inputs=[image, style, custom, text, control, strength, steps, guidance, remove_bg, position, seed],
            outputs=[artwork, contours, status], concurrency_limit=1,
            concurrency_id="local-gpu", show_progress="full", trigger_mode="once",
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    pipeline.warmup()
    build_ui().launch(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="stone"),
        css=CSS, max_file_size="20mb", footer_links=[],
    )


if __name__ == "__main__":
    main()
