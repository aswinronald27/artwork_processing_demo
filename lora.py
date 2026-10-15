"""Small, dependency-free LoRA for SD 1.5 attention projections.

Artifacts use the ThreadOps native format, not the PEFT/Diffusers checkpoint
format. Base weights are never included or modified in saved adapters.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.multiplier = alpha / rank
        self.strength = 1.0

    def forward(self, value):
        result = self.base(value)
        if self.strength == 0:
            return result
        # Frozen FP16 base, trainable FP32 adapters; gradients remain FP32.
        with torch.autocast(device_type=value.device.type, enabled=False):
            update = self.up(self.down(value.float()))
        return result + update.to(result.dtype) * self.multiplier * self.strength


def attach_lora(unet, rank=8, alpha=8):
    adapters = {}
    for name, module in list(unet.named_modules()):
        if isinstance(module, LoRALinear):
            raise ValueError("Model already contains a native LoRA adapter")
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in (
            ".to_q", ".to_k", ".to_v", ".to_out.0",
        )):
            parent_name, child_name = name.rsplit(".", 1)
            adapter = LoRALinear(module, rank, alpha)
            setattr(unet.get_submodule(parent_name), child_name, adapter)
            adapters[name] = adapter
    if not adapters:
        raise ValueError("No attention projection layers found")
    return adapters


def save_adapter(adapters, path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, module in adapters.items():
        weights[name + ".down.weight"] = module.down.weight.detach().float().cpu().contiguous()
        weights[name + ".up.weight"] = module.up.weight.detach().float().cpu().contiguous()
    temp = path.with_suffix(".tmp")
    save_file(weights, str(temp), metadata={"format": "threadops_lora_v1", "config": json.dumps(config)})
    temp.replace(path)


def load_adapter(unet, path, base_model=None):
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata()
    if meta.get("format") != "threadops_lora_v1":
        raise ValueError("Expected a ThreadOps native LoRA checkpoint")
    config = json.loads(meta["config"])
    if base_model is not None and config["base_model"] != base_model:
        raise ValueError(f"Adapter requires {config['base_model']}, not {base_model}")
    weights = load_file(str(path), device="cpu")
    # Validate all keys/shapes before mutating the model.
    targets = {name: module for name, module in unet.named_modules()
               if isinstance(module, nn.Linear) and any(name.endswith(s) for s in (".to_q", ".to_k", ".to_v", ".to_out.0"))}
    expected = {}
    for name, module in targets.items():
        expected[name + ".down.weight"] = (config["rank"], module.in_features)
        expected[name + ".up.weight"] = (module.out_features, config["rank"])
    if set(weights) != set(expected) or any(tuple(weights[k].shape) != shape for k, shape in expected.items()):
        raise ValueError("Adapter attention keys/shapes do not match this model")
    adapters = attach_lora(unet, config["rank"], config["alpha"])
    with torch.no_grad():
        for name, module in adapters.items():
            module.down.weight.copy_(weights[name + ".down.weight"])
            module.up.weight.copy_(weights[name + ".up.weight"])
    return adapters, config
