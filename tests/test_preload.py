import unittest
from unittest.mock import Mock, patch

import pipeline
import preload_models


class PreloadTests(unittest.TestCase):
    def test_downloads_only_half_precision_weights_serially_then_verifies(self):
        with patch.object(pipeline.torch.cuda, "is_available", return_value=True):
            with patch("huggingface_hub.snapshot_download") as download:
                with patch.object(pipeline, "load_diffusion_models") as verify:
                    with patch.object(pipeline, "create_rembg_session") as background:
                        preload_models.preload()
        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["max_workers"], 1)
            weights = [p for p in call.kwargs["allow_patterns"] if "safetensors" in p]
            self.assertTrue(weights)
            self.assertTrue(all("fp16.safetensors" in p for p in weights))
            self.assertFalse(any(".bin" in p for p in call.kwargs["allow_patterns"]))
        verify.assert_called_once()
        background.assert_called_once_with(local_files_only=False)

    def test_local_loader_requires_fp16_safetensors_and_never_downloads(self):
        with patch("diffusers.ControlNetModel.from_pretrained") as control:
            with patch("diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained") as base:
                pipeline.load_diffusion_models()
        for loader in (control, base):
            self.assertEqual(loader.call_args.kwargs["torch_dtype"], pipeline.torch.float16)
            self.assertEqual(loader.call_args.kwargs["variant"], "fp16")
            self.assertTrue(loader.call_args.kwargs["use_safetensors"])
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertIsNone(base.call_args.kwargs["safety_checker"])

    def test_u2netp_explicitly_selects_cpu_provider(self):
        with patch("rembg.new_session") as create:
            pipeline.create_rembg_session(local_files_only=False)
        create.assert_called_once_with(model_name="u2netp", providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
