"""Local, resumable SD 1.5 style LoRA training on the prepared captioned art.

Cache frozen VAE latents/text embeddings first, then keep only the UNet and
small FP32 adapters on GPU. No model downloads, telemetry, or PEFT dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline  # Set local cache paths before Hugging Face imports.
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from lora import attach_lora, load_adapter, save_adapter


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def read_dataset(folder):
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50:
        raise ValueError(f"Expected 50 curated training artworks, found {len(rows)}")
    digest = hashlib.sha256()
    for row in rows:
        image_path = (folder / row["file_name"]).resolve()
        if not image_path.is_relative_to(folder.resolve()):
            raise ValueError("Dataset image escapes training folder")
        digest.update(image_path.read_bytes())
        digest.update(row["text"].encode())
        with Image.open(image_path) as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError(f"Expected 512px artwork: {image_path}")
    return rows, digest.hexdigest()


def cache_training_inputs(rows, folder, base_model, output, dataset_hash):
    cache_file = output / "training-cache.pt"
    if cache_file.is_file():
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        if cached["base_model"] == base_model and cached["dataset_hash"] == dataset_hash:
            print("[Cache] Reusing validated latents and captions", flush=True)
            return cached["latents"], cached["embeddings"]
    options = dict(variant="fp16", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", **options)
    encoder.requires_grad_(False).eval().to("cuda")
    embeddings = []
    print("[Cache] Encoding the 50 training captions", flush=True)
    with torch.no_grad():
        for row in rows:
            tokens = tokenizer(row["text"], padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            embeddings.append(encoder(tokens)[0].cpu())
    del encoder
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", **options)
    vae.requires_grad_(False).eval().to("cuda")
    latents = []
    print("[Cache] Encoding the 50 training artworks (batch 1)", flush=True)
    with torch.no_grad():
        for index, row in enumerate(rows):
            with Image.open(folder / row["file_name"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
            pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)
            latent = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            if not torch.isfinite(latent).all():
                raise RuntimeError("VAE produced nonfinite latents")
            latents.append(latent.cpu())
            if (index + 1) % 10 == 0:
                print(f"[Cache] {index + 1}/50 images encoded", flush=True)
    del vae
    torch.cuda.empty_cache()
    latents, embeddings = torch.cat(latents), torch.cat(embeddings)
    torch.save(dict(base_model=base_model, dataset_hash=dataset_hash, latents=latents, embeddings=embeddings), cache_file)
    return latents, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--base-model", default="Lykon/dreamshaper-8")
    parser.add_argument("--output", type=Path, default=ROOT / "training/checkpoints/threadops-flat")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.rank < 1 or args.save_every < 1 or not 0 < args.lr <= 0.01:
        parser.error("Invalid steps, rank, save interval, or learning rate")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # One owner per output prevents overlapping runs corrupting checkpoints.
    lock_file = output / "training.lock"
    try:
        lock = lock_file.open("x")
    except FileExistsError as exc:
        raise RuntimeError(f"Training lock exists at {lock_file}; check the existing run before removing it.") from exc
    lock.write(str(os.getpid()))
    lock.close()
    status_path = output / "status.json"
    status = {"state": "initializing", "pid": os.getpid(), "step": 0, "total_steps": args.steps}
    config = None
    adapters = None
    optimizer = None
    scheduler = None
    scaler = None
    step = 0
    started = time.time()

    def checkpoint(label):
        folder = output / label
        folder.mkdir(exist_ok=True)
        saved_config = dict(config, step=step)
        adapter_path = folder / "threadops_lora.safetensors"
        save_adapter(adapters, adapter_path, saved_config)
        temp = folder / "training-state.tmp"
        torch.save({
            "step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
            "config": config,
        }, temp)
        temp.replace(folder / "training-state.pt")
        write_json(output / "latest.json", {"step": step, "folder": str(folder), "adapter": str(adapter_path)})
        print(f"[Checkpoint] {adapter_path}", flush=True)

    try:
        torch.manual_seed(42)
        random.seed(42)
        dataset = ROOT / "training/prepared/pets/train"
        rows, dataset_hash = read_dataset(dataset)
        config = dict(base_model=args.base_model, rank=args.rank, alpha=args.rank,
                      learning_rate=args.lr, total_steps=args.steps, dataset_hash=dataset_hash,
                      training_images=len(rows), resolution=512, trigger="threadopsflat", seed=42)
        if (output / "run-config.json").is_file() and not args.resume:
            raise RuntimeError("This output already has a run; use --resume or a different --output folder.")
        if args.resume and (output / "run-config.json").is_file():
            old = json.loads((output / "run-config.json").read_text())
            if old != config:
                raise RuntimeError("Resume configuration/dataset differs from the existing run")
        write_json(output / "run-config.json", config)
        status.update(base_model=args.base_model, training_images=len(rows))
        write_json(status_path, status)
        print(f"[GPU] {torch.cuda.get_device_name(0)} | {len(rows)} images | rank {args.rank}", flush=True)
        latents, embeddings = cache_training_inputs(rows, dataset, args.base_model, output, dataset_hash)
        print("[Model] Loading frozen FP16 UNet; training attention LoRA weights only", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model, subfolder="unet", variant="fp16", torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True,
        ).requires_grad_(False)
        resume_state = None
        if args.resume and (output / "latest.json").is_file():
            latest = json.loads((output / "latest.json").read_text())
            adapters, _ = load_adapter(unet, latest["adapter"], args.base_model)
            resume_state = torch.load(Path(latest["folder"]) / "training-state.pt", map_location="cpu", weights_only=True)
        else:
            adapters = attach_lora(unet, args.rank, args.rank)
        unet.enable_gradient_checkpointing()
        unet.train().to("cuda")
        # .to(device) preserves FP32 adapter weights and FP16 frozen base weights.
        trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        print(f"[Train] {sum(p.numel() for p in trainable):,} trainable parameters; batch 1; gradient checkpointing", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

        def learning_rate_scale(index):
            warmup = min(50, max(1, args.steps // 10))
            return min(1.0, (index + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * index / args.steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
        scaler = torch.amp.GradScaler("cuda")
        noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler", local_files_only=True)
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])
            step = resume_state["step"]
            torch.set_rng_state(resume_state["torch_rng"])
            torch.cuda.set_rng_state(resume_state["cuda_rng"])
            random.setstate(resume_state["python_rng"])
            print(f"[Resume] Continuing after step {step}", flush=True)
        training_started = time.time()
        initial_step = step
        recent_losses = []
        consecutive_overflows = 0
        while step < args.steps:
            index = random.randrange(len(rows))
            clean = latents[index:index+1].to("cuda")
            text = embeddings[index:index+1].to("cuda")
            noise = torch.randn_like(clean)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda")
            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                predicted = unet(noisy, timesteps, encoder_hidden_states=text).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean, noise, timesteps)
                else:
                    raise ValueError("Unsupported noise prediction type")
                loss = F.mse_loss(predicted.float(), target.float())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; stopped without overwriting the last good checkpoint")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(gradient_norm):
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                consecutive_overflows += 1
                if consecutive_overflows >= 20:
                    raise RuntimeError("Repeated nonfinite gradients; stopped after 20 AMP retries")
                print("[AMP] Nonfinite gradients; reduced scaler and retrying", flush=True)
                continue
            consecutive_overflows = 0
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            recent_losses.append(float(loss.detach()))
            if step == 1 or step % 10 == 0 or step == args.steps:
                seconds_per_step = (time.time() - training_started) / max(1, step - initial_step)
                status.update(state="training", step=step, loss=sum(recent_losses)/len(recent_losses),
                              seconds_per_step=seconds_per_step, eta_seconds=(args.steps-step)*seconds_per_step,
                              elapsed_seconds=time.time()-started, gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                              updated_at=time.time())
                write_json(status_path, status)
                print(f"[Step {step}/{args.steps}] loss={status['loss']:.5f} | {seconds_per_step:.2f}s/step | ETA {status['eta_seconds']/60:.1f}min | peak {status['gpu_peak_gb']:.2f}GB", flush=True)
                recent_losses.clear()
            if step % args.save_every == 0:
                checkpoint(f"step-{step:04d}")
        checkpoint("final")
        status.update(state="completed", step=step, elapsed_seconds=time.time()-started)
        write_json(status_path, status)
        print("[Complete] Adapter saved. Visual validation is required before use in production.", flush=True)
    except KeyboardInterrupt:
        if adapters is not None and optimizer is not None and step > 0:
            checkpoint(f"interrupted-{step:04d}")
        status.update(state="interrupted", step=step)
        write_json(status_path, status)
        raise
    except Exception as exc:
        status.update(state="failed", step=step, error=str(exc))
        write_json(status_path, status)
        raise
    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
