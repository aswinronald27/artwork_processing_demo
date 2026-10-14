# ThreadOps AI Art Studio

Local photo-to-art demo for Windows 11, Python 3.12, and an NVIDIA RTX 3050 with 8 GB VRAM.

Double-click **start_demo.bat**. It creates `.venv` if needed, installs pinned dependencies, checks CUDA, pre-downloads and verifies models, then serves **http://127.0.0.1:7860**. Open that URL manually. Alternatively:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe preload_models.py
.venv\Scripts\python.exe app.py
```

Select a sample or upload/paste/capture a photo, choose a style, optionally enter a name, then click **Generate Digitizer Art**. Style changes reset likeness and transfer strength to that preset's defaults. Advanced settings include name placement and a seed; reuse the same seed and inputs to compare strengths. Actual diffusion passes equal approximately steps × strength, as expected for image-to-image.

## Models and local operation

Run `preload_models.py` before starting the app. It downloads only the FP16 safetensors variants of SD 1.5 and Canny ControlNet, plus the lightweight rembg U2NetP model. Downloads use one worker to avoid parallel multi-gigabyte transfers. Clear console stages and download bars show setup progress; failure exits with a nonzero status and the launcher does not start the server. Downloads default to `.models/` inside this project. Allow several GB of disk space and enough system RAM for CPU offload.

`app.py` calls `warmup()` before binding the server. Warmup reads cached FP16 weights with `local_files_only=True`, enables memory optimizations, and initializes the CPU-only background session. The Generate handler only accesses warmed models; it cannot start downloads or model initialization. Both the diffusion step callback and Gradio's tracked progress provide live feedback from 0% to 100%. Inference and photos remain local; no cloud inference endpoint or paid API is used.

The default backbone is `stable-diffusion-v1-5/stable-diffusion-v1-5`, a mirror of the deprecated Runway repository (the original currently returns an authorization error). The loader explicitly requires `variant="fp16"`, safetensors, and `torch.float16`, then enables attention/VAE slicing and model CPU offload. rembg uses `new_session(model_name="u2netp", providers=["CPUExecutionProvider"])` to avoid loading incompatible CUDA provider DLLs and keep VRAM available for diffusion. The safety checker is disabled as requested.

After successful preloading, no internet is needed for the app. Verify your cache offline:

```powershell
$env:THREADOPS_OFFLINE = "1"
.venv\Scripts\python.exe preload_models.py --offline
.venv\Scripts\python.exe app.py
```

If diffusion weights are missing, startup fails immediately with preloader instructions. If background weights cannot initialize, startup reports it, and background removal falls back to the original on white without trying another download. `HF_HUB_OFFLINE=1` may additionally be set for Hugging Face's own offline mode.

Optional configuration (set before launching):

```powershell
# SD 1.5 Diffusers model with FP16 variant files, or a local folder with those files:
$env:THREADOPS_MODEL_ID = "Lykon/dreamshaper-8"
# Optional alternate Hugging Face / rembg caches:
$env:HF_HOME = "C:\models\huggingface"
$env:U2NET_HOME = "C:\models\rembg"
```

A custom varsity TrueType font may be placed at `assets/varsity.ttf`. Otherwise Windows Arial Black/Bold is used, with a portable bold fallback. Name banners shrink to fit and support top/bottom placement.

## Practical limits

Outputs are 512 × 512 PNG raster artwork, including vector-*style* illustrations and simulated thread texture. They are reference artwork for a human digitizer, not SVG vectors or machine-ready DST/PES stitch files. ControlNet guides silhouettes; it cannot guarantee exact identity or removal of all facial features. Check generated artwork before production.

Background isolation failures are reported without stopping generation. If CUDA is unavailable, the app reports that fact and generation fails with setup guidance. GPU memory errors suggest closing other GPU apps. Generation is serialized because the pipeline's offload hooks share GPU state. The launcher binds only to localhost and disables Gradio sharing and analytics.

## Validation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover FP16-only serial downloads, cache-only startup, request-time loading prevention, CPU provider selection, progress callbacks, source/contour separation, denoise and seed wiring, transparency and padding, banner fitting, and UI wiring. Run a real GPU smoke test after preloading:

```powershell
.venv\Scripts\python.exe verify_gpu.py
```

This warms the cached models, generates a collie portrait, and writes `outputs/gpu-smoke-artwork.png` and `outputs/gpu-smoke-contours.png`. Model downloads and CUDA are required for this integration check.

References: [PyTorch CUDA 12.4 wheels](https://docs.pytorch.org/get-started/previous-versions/), [ControlNet image-to-image API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet), [SD 1.5 mirror/model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [rembg](https://github.com/danielgatis/rembg).
