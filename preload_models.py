"""Serial FP16-only model downloads, before the Gradio server starts."""
from __future__ import annotations

import argparse
import gc
import sys

import pipeline


def preload(local_files_only: bool = False):
    if not pipeline.torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU not detected. Install requirements.txt and check nvidia-smi.")
    print("[GPU] " + pipeline.device_description(), flush=True)
    print("[Cache] " + str(pipeline.os.environ["HF_HOME"]), flush=True)
    from huggingface_hub import snapshot_download

    print("[1/4] ControlNet Canny: FP16 safetensors only (one download at a time)", flush=True)
    snapshot_download(
        repo_id=pipeline.CONTROLNET_ID,
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors" if pipeline.IS_SDXL else "diffusion_pytorch_model.fp16.safetensors"],
        max_workers=1, local_files_only=local_files_only,
    )
    print(f"[2/4] {pipeline.MODEL_ID} backbone: FP16 safetensors only; safety checker excluded", flush=True)
    if not pipeline.Path(pipeline.MODEL_ID).is_dir():
        snapshot_download(
            repo_id=pipeline.MODEL_ID,
            allow_patterns=[
                "model_index.json", "scheduler/scheduler_config.json", "tokenizer/*",
                "feature_extractor/preprocessor_config.json",
                "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                *(["tokenizer_2/*", "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors"] if pipeline.IS_SDXL else []),
                "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
            ],
            max_workers=1, local_files_only=local_files_only,
        )
    if pipeline.STYLE_LORA_ID:
        print(f"[Style] Downloading/verifying {pipeline.STYLE_LORA_ID}", flush=True)
        snapshot_download(repo_id=pipeline.STYLE_LORA_ID,
                          allow_patterns=[pipeline.STYLE_LORA_FILE, "README.md"],
                          max_workers=1, local_files_only=local_files_only)
        from faceless import preload_face_model
        preload_face_model(local_files_only=local_files_only)
    print("[3/4] Verifying local diffusion models (torch.float16 / variant=fp16)", flush=True)
    model = pipeline.load_diffusion_models()
    del model
    gc.collect()
    print(f"[4/4] Downloading/verifying {pipeline.REMBG_MODEL} (CPUExecutionProvider)", flush=True)
    session = pipeline.create_rembg_session(local_files_only=local_files_only)
    del session
    print("[Complete] All weights verified. Ready to launch python app.py", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify existing cache without network access")
    args = parser.parse_args()
    try:
        preload(local_files_only=args.offline or pipeline.OFFLINE)
    except Exception as exc:
        print(f"[Preload failed] {exc}\nServer was not started. Fix connectivity/cache and rerun preload_models.py.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
