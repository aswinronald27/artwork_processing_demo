# ThreadOps portrait style dataset

## Current prepared set

The uploaded 50 photo/art pairs have been reviewed. The pet pilot now contains **50 captioned standalone artworks: 18 cats and 32 dogs**, at `training/prepared/pets/train/`. This includes 15 selected artworks from `Demo Images` and 35 visually selected additions from the authorized Work folder's JPG/JPEG files. Three original pet photo/art pairs remain held out. Additional artwork is unpaired unless a source photo is explicitly mapped; no pair is inferred from a nearby filename.

The JPG/JPEG scan found 4,307 files and recorded zero read errors. White-background filtering narrowed the review; only selected pet portraits were copied into `training/data/work-artwork/`. Original files were left unchanged. A geometric/color similarity check compares additions against prepared training and holdout art. It does not prove that different photos depict different pets; the owner should confirm identities across the final split before treating evaluation as independent.

`training/supplement_selection.json` records selection, source paths, species, and content descriptions. `training/prepared/manifest.json` records source roles, review flags, split assignments, and captions. `metadata.jsonl` is the authoritative list of training files. Prepared review sheets are under `outputs/dataset-audit/expanded-training-1.jpg` and `expanded-training-2.jpg`.

Rebuild with `.venv\Scripts\python.exe training/prepare_dataset.py`. Selected local copies are reused. This command only prepares and verifies data; it does not train a model.

## Local training and progress

Run `start_training.bat` from the project root to train or resume the pet style pilot. It uses the locally cached DreamShaper 8 backbone, 50 captioned training artworks, resolution 512, batch size 1, rank 8 attention adapters, FP16 frozen base weights, FP32 adapter parameters, gradient checkpointing, and AdamW at a maximum learning rate of 1e-4. The pilot uses 1,000 optimizer steps, saves every 100 steps, and does not load the held-out photos/artworks.

PEFT could not be installed through this session's restricted network, so `lora.py` implements the small low-rank attention updates directly in PyTorch. Native artifacts are explicitly tagged `threadops_lora_v1`; they are **not** advertised as PEFT/Diffusers adapter files. The custom loader is included in this app. Checkpoint math, frozen base parameters, and save/load round trips are covered by tests. No model download or cloud upload occurs during training.

The live status is `training/checkpoints/threadops-flat/status.json` (state, PID, step, loss, GPU memory, elapsed time, ETA). Logs from the agent-started run are in `training-run.log`. Checkpoints include small adapter safetensors and optimizer/scaler/RNG state. `--resume` continues from `latest.json`; an output lock prevents overlapping runs. Do not launch a second trainer into the same output directory.

After completion, `start_trained_demo.bat` loads the final adapter onto its matching base model. The adapter applies only to the flat pet preset; faceless/embroidery presets disable its contribution. Training completion is **not** a claim of acceptable output quality. Evaluate the reserved photos, inspect likeness and flat shapes, and compare intermediate checkpoints before choosing a production model. The existing holdout covers dogs; cat generalization still needs independent cat photo/art pairs.

Training may take many minutes and is separate from the per-image inference budget. The measured three-step smoke test used roughly 1.9 GB of allocated GPU memory on the RTX 3050; use the live status for actual run speed and ETA.

The current SD 1.5 / DreamShaper demo has not achieved the requested clean vector portrait style. A dedicated style LoRA should be evaluated on held-out photos before any quality claim. Training is separate from generation: allowing 30 seconds per generated image does not imply training completes in 30 seconds.

## What to supply

Start with 20–50 carefully selected finished artworks in one consistent style. This is a pilot target, not a guaranteed minimum or a promise of quality. More varied, consistent examples can help. Avoid mixing painterly portraits, embroidery textures, faceless couples, or clothing mockups into the flat pet dataset.

Include cats and dogs if both must be supported; vary fur length, ear shape, coat colors, and head angle. Supply standalone artwork files at their original resolution, ideally at least 512 pixels per side, without banners or watermarks. Keep stylistic consistency more important than quantity. Do not resize or crop the originals manually for training yet.

Matching raw customer photos are valuable for evaluating likeness. A style LoRA can train on captioned finished artwork without paired photos; it does not, by itself, learn a paired photo-to-art conversion. Likeness still needs image conditioning and validation.

## Folder structure

```
training/data/artwork/pet_001.png
training/data/artwork/pet_002.png
training/data/photos/pet_001.jpg
training/data/photos/pet_002.jpg
training/data/holdout/photos/pet_101.jpg
training/data/holdout/expected/pet_101.png
```

Use the same filename stem for matching photos and artwork. A matched photo is optional for the training artwork. Keep around 5–10 additional photo/art pairs aside for evaluation, if available; these must not be used for training. Different images of the same pet should stay in one split to avoid misleading validation.

For a first inspection, 5 representative pairs are enough to audit the style and source framing before preparing the full dataset. Provide the local folder path once populated. Files in `training/data/` are ignored by Git.

## Captions and evaluation

Each training artwork will need a caption describing its content (species, coat markings, pose), with one consistent style trigger. For example: `threadopsflat, a black tortoiseshell cat head portrait, amber eyes, white chin, white background`. Describe changing content so the model can separate species/identity from style. Avoid labeling every example as a dog or forcing a breed not present in the image.

Evaluate style and likeness separately: crisp flat color regions, controlled edges, pure white background, correct species, coat pattern, eye placement, ear shape, and absence of invented limbs or borders. Use the same seed and input when comparing checkpoints. Do not select a checkpoint using training examples alone.

## Intended local training approach

Use an SD 1.5-compatible LoRA first, with a frozen base model and small trainable adapter weights. Begin at 512px, batch size 1, FP16, and gradient checkpointing; actual VRAM usage must be measured on this RTX 3050. Close the inference server while training. Training speed and final settings will be determined after inspecting the dataset; no training has started.

Reference: [Hugging Face Diffusers LoRA training](https://huggingface.co/docs/diffusers/training/lora).
