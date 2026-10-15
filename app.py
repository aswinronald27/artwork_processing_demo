"""Gradio dashboard for ThreadOps; only binds to localhost."""
from __future__ import annotations

import os
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import pipeline
import gemini_service
import gradio as gr

STYLES = list(pipeline.STYLE_PROMPTS)
CLOUD_ENGINE = "Cloud (Gemini Nano Banana 2.1)"
PET_PRESET = pipeline.STYLE_PROMPTS[STYLES[0]]
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


def preview_subject(image):
    if image is None:
        raise gr.Error("Upload a customer photo first.")
    notices = []
    prepared = pipeline.preprocess(image, True, notices)
    for message in notices:
        gr.Warning(message)
    return prepared


def enhance_prompt(image, style, api_key):
    try:
        return gemini_service.analyze_and_prompt(image, style, api_key)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_generation(image, style, custom_prompt, text_overlay, control_scale,
                   denoise_strength, steps, guidance_scale, remove_bg,
                   text_position="Bottom", seed=-1, api_key=None,
                   engine_choice="Hybrid (Gemini Vision + Local GPU)",
                   face_cleanup=True,
                   progress=gr.Progress(track_tqdm=True)):
    """Stream status; Gradio's progress display supplies a live elapsed timer."""
    if image is None:
        yield None, None, "**Photo needed.** Upload an image or select a sample below."
        return
    started = time.perf_counter()
    use_cloud = engine_choice in (CLOUD_ENGINE, "Cloud (Google Imagen 3)") and bool(
        (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    )
    if use_cloud:
        yield None, None, "**Generating with Gemini Nano Banana 2.1…**"
        try:
            full_prompt = pipeline.STYLE_PROMPTS[style]["prompt"]
            if custom_prompt.strip():
                full_prompt += ", " + custom_prompt.strip()
            full_prompt = (
                "Use the supplied photo as the reference for the subjects, clothing, colors, "
                "markings, and pose. Transform it into the requested artwork style. "
                "Return only the finished square illustration without added text. " + full_prompt
            )
            artwork = gemini_service.generate_with_gemini(prompt=full_prompt, api_key=api_key, image=image)
            if text_overlay:
                artwork = pipeline.add_text_overlay(artwork, text_overlay, position=text_position)
            elapsed = time.perf_counter() - started
            yield artwork, None, f"**Completed in {elapsed:.1f}s** · {artwork.width} × {artwork.height} PNG  \nGemini Nano Banana 2.1"
        except Exception as exc:
            gr.Warning(str(exc))
            yield None, None, f"**Generation failed after {time.perf_counter() - started:.1f}s.** See the error notification and terminal for details."
        return
    yield None, None, "**Generating…** " + pipeline.device_description() + "  \nFollow the live diffusion progress and elapsed timer above."
    notices = []
    try:
        contours, artwork = pipeline.generate(
            input_image=image, style_choice=style, custom_prompt=custom_prompt,
            text_overlay=text_overlay, control_strength=control_scale,
            denoise_strength=denoise_strength, steps=steps,
            guidance_scale=guidance_scale, isolate_background=remove_bg,
            text_position=text_position, seed=seed, notices=notices,
            simplify_face_features=face_cleanup,
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
    return [[str(samples / row[0]), *row[1:4], *on_style_change(row[1])]
            for row in rows if (samples / row[0]).is_file()]


def build_ui():
    with gr.Blocks(title="ThreadOps AI Art Studio (Local GPU)", analytics_enabled=False) as demo:
        gr.Markdown(
            "# ThreadOps AI Art Studio (Local GPU)\n"
            "Customer photo → apparel artwork. Three art treatments, precise contour guidance, and a personal name banner. "
            "**Local GPU or optional Google cloud generation.**", elem_id="studio-header",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"],
                                 label="Customer photo", height=300, format="png", image_mode="RGBA")
                enhance_btn = gr.Button(" Auto-Describe Photo with Gemini", size="sm")
                with gr.Accordion("Check subject framing", open=False):
                    preview_button = gr.Button("Preview isolated subject")
                    prepared_preview = gr.Image(label="Prepared subject", interactive=False, format="png", height=260)
                    gr.Markdown("Check both ears and the coat outline before generating. A tightly cropped input helps when background isolation misses details.")
                style = gr.Dropdown(STYLES, value=STYLES[0], label="Art style")
                custom = gr.Textbox(label="Custom prompt additions", placeholder="e.g. older child holding a baby, pale yellow dresses, black hair", lines=2,
                                    info="For local generation, use a brief visual description. Long instruction lists exceed the model's 77-token limit, including the selected style.")
                api_key = gr.Textbox(label="Gemini API Key", placeholder="Paste AI Studio API key (AIzaSy...)",
                                     type="password", value=os.getenv("GEMINI_API_KEY", ""))
                engine_choice = gr.Radio(choices=["Hybrid (Gemini Vision + Local GPU)", CLOUD_ENGINE],
                                         value="Hybrid (Gemini Vision + Local GPU)", label="Generation Engine")
                text = gr.Textbox(label="Name banner (optional)", placeholder="OREO / MUMMY & DADDY", max_length=80)
                with gr.Accordion("Advanced settings", open=False):
                    remove_bg = gr.Checkbox(value=True, label="Isolate and enlarge subject on white (rembg)")
                    control = gr.Slider(0.1, 1.5, value=PET_PRESET["default_control"], step=0.01, label="Contour strictness / likeness")
                    strength = gr.Slider(0.2, 1.0, value=PET_PRESET["default_denoise"], step=0.01, label="Style transfer strength",
                                         info="Higher values redraw more of the original photo.")
                    steps = gr.Slider(20, 30, value=25, step=1, label="Inference steps")
                    guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Prompt guidance")
                    position = gr.Radio(["Top", "Bottom"], value="Bottom", label="Name placement")
                    seed = gr.Number(value=-1, precision=0, minimum=-1, maximum=4294967295, label="Seed (-1 = random)")
                    face_cleanup = gr.Checkbox(value=True, label="Simplify detected faces", visible=pipeline.IS_SDXL,
                                                info="For the local SDXL faceless style. May affect glasses or face accessories; turn off if needed.")
                generate = gr.Button("Generate Digitizer Art", variant="primary", size="lg")
            with gr.Column(scale=6, min_width=320):
                status = gr.Markdown("**Ready** · " + pipeline.device_description() + f"  \nBackbone: `{pipeline.MODEL_ID}` · isolation: `{pipeline.REMBG_MODEL}`", elem_id="studio-status")
                with gr.Row():
                    artwork = gr.Image(label="Generated Artwork", type="pil", format="png", interactive=False, height=460)
                    contours = gr.Image(label="Detected Contours (ControlNet Canny Map)", type="pil", format="png", interactive=False, height=460)
                gr.Markdown("Download the PNG from its preview. Artwork is a **raster reference for your digitizer**; embroidery stitch files and SVG paths require a separate digitizing step.")
        examples = example_rows()
        if examples:
            gr.Examples(examples=examples, inputs=[image, style, custom, text, control, strength],
                        label="Sample photos · click to load, then Generate", cache_examples=False)
        gr.Markdown("Local models are preloaded before startup. Local generation works without an API key. "
                    "Auto-Describe uses Gemini 3.8 Flash. Cloud sends the photo and prompt to Gemini Nano Banana 2.1 "
                    "and may incur API charges. An empty key falls back to local generation.")
        enhance_btn.click(enhance_prompt, inputs=[image, style, api_key], outputs=custom)
        style.change(on_style_change, inputs=style, outputs=[control, strength], queue=False)
        style.change(lambda value: pipeline.STYLE_PROMPTS[value]["default_guidance"],
                     inputs=style, outputs=guidance, queue=False)
        preview_button.click(preview_subject, inputs=image, outputs=prepared_preview,
                             concurrency_limit=1, concurrency_id="local-gpu")
        generate.click(
            run_generation,
            inputs=[image, style, custom, text, control, strength, steps, guidance, remove_bg, position, seed, api_key, engine_choice, face_cleanup],
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
