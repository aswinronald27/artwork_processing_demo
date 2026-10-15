import unittest
from pathlib import Path

import torch
from torch import nn

from lora import attach_lora, load_adapter, save_adapter


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(3, 4)

    def forward(self, inputs):
        return self.attn.to_q(inputs)


class NativeLoRATests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_base_and_freezes_its_weights(self):
        model = TinyAttention().requires_grad_(False)
        inputs = torch.randn(2, 3)
        before = model(inputs).detach()
        adapters = attach_lora(model, rank=2, alpha=2)
        self.assertTrue(torch.equal(before, model(inputs)))
        model(inputs).square().mean().backward()
        adapter = adapters["attn.to_q"]
        self.assertIsNone(adapter.base.weight.grad)
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0)

    def test_adapter_math_and_strength(self):
        model = TinyAttention().requires_grad_(False)
        inputs = torch.randn(2, 3)
        adapter = attach_lora(model, rank=2, alpha=4)["attn.to_q"]
        with torch.no_grad():
            adapter.up.weight.fill_(0.2)
        expected = adapter.base(inputs) + 2 * adapter.up(adapter.down(inputs))
        self.assertTrue(torch.allclose(model(inputs), expected))
        adapter.strength = 0
        self.assertTrue(torch.equal(model(inputs), adapter.base(inputs)))

    def test_checkpoint_roundtrip_and_base_model_validation(self):
        model = TinyAttention().requires_grad_(False)
        other = TinyAttention().requires_grad_(False)
        other.load_state_dict(model.state_dict())
        adapter = attach_lora(model, rank=2, alpha=2)
        with torch.no_grad():
            adapter["attn.to_q"].up.weight.normal_()
        path = Path(__file__).resolve().parents[1] / "outputs" / "native-lora-test.safetensors"
        path.parent.mkdir(exist_ok=True)
        save_adapter(adapter, path, {"base_model": "test-base", "rank": 2, "alpha": 2})
        with self.assertRaisesRegex(ValueError, "requires test-base"):
            load_adapter(other, path, "wrong-base")
        load_adapter(other, path, "test-base")
        inputs = torch.randn(4, 3)
        self.assertTrue(torch.equal(model(inputs), other(inputs)))


if __name__ == "__main__":
    unittest.main()
