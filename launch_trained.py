"""Launch the finished native LoRA; never substitute an unfinished checkpoint."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
folder = ROOT / "training/checkpoints/threadops-flat"
adapter = folder / "final/threadops_lora.safetensors"
status = json.loads((folder / "status.json").read_text()) if (folder / "status.json").is_file() else {}
if status.get("state") != "completed" or not adapter.is_file():
    raise SystemExit("Training is not complete yet. Check training/checkpoints/threadops-flat/status.json.")
config = json.loads((folder / "run-config.json").read_text())
os.environ["THREADOPS_MODEL_ID"] = config["base_model"]
os.environ["THREADOPS_LORA_PATH"] = str(adapter)
os.environ["THREADOPS_REMBG_MODEL"] = "u2net"

if __name__ == "__main__":
    import app
    app.main()
