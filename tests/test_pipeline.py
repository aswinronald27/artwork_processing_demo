import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

import pipeline


class PreprocessingTests(unittest.TestCase):
    def test_small_subject_fills_canvas_without_background_or_mask_speckles(self):
        cutout = Image.new("RGBA", (600, 1200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cutout)
        draw.rectangle((250, 650, 350, 850), fill=(180, 80, 20, 255))
        draw.point((0, 0), fill=(0, 0, 0, 255))
        framed = pipeline.frame_cutout(cutout)
        foreground = np.any(np.asarray(framed) < 245, axis=2)
        ys, xs = np.nonzero(foreground)
        self.assertGreater(ys.max() - ys.min(), 430)
        self.assertGreater(xs.min(), 130)
        self.assertEqual(framed.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(framed.getpixel((256, 256)), (180, 80, 20))

    def test_subject_crop_keeps_multiple_people(self):
        cutout = Image.new("RGBA", (1000, 1000))
        draw = ImageDraw.Draw(cutout)
        draw.rectangle((100, 400, 200, 700), fill="red")
        draw.rectangle((500, 400, 600, 700), fill="blue")
        pixels = np.asarray(pipeline.frame_cutout(cutout))
        self.assertTrue(np.any(np.all(pixels == [255, 0, 0], axis=2)))
        self.assertTrue(np.any(np.all(pixels == [0, 0, 255], axis=2)))

    def test_padding_preserves_aspect_ratio_and_transparency(self):
        source = Image.new("RGBA", (400, 200), (255, 0, 0, 255))
        result = pipeline.preprocess(source, False)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.getpixel((256, 20)), (255, 255, 255))
        self.assertEqual(result.getpixel((256, 256)), (255, 0, 0))
        transparent = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
        self.assertEqual(pipeline.preprocess(transparent, False).getpixel((256, 256)), (255, 255, 255))

    def test_canny_is_rgb_with_black_background_and_white_outline(self):
        image = Image.new("RGB", (512, 512), "white")
        ImageDraw.Draw(image).rectangle((100, 100, 400, 400), fill="black")
        edges = pipeline.extract_canny_edges(image)
        self.assertEqual(edges.mode, "RGB")
        pixels = np.asarray(edges)
        self.assertEqual(set(np.unique(pixels)), {0, 255})
        self.assertGreater(np.count_nonzero(pixels), 100)

    def test_offline_missing_rembg_weights_falls_back_visibly(self):
        cache = str(pipeline.ROOT / "tests" / "missing-rembg-cache")
        with patch.object(pipeline, "OFFLINE", True), patch.dict(os.environ, {"U2NET_HOME": cache}):
            notices = []
            result = pipeline.remove_background(Image.new("RGBA", (40, 40), (0, 0, 0, 0)), notices)
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertIn("Background removal unavailable", notices[0])

    def test_long_name_fits_and_placement_does_not_change_other_end(self):
        source = Image.new("RGB", (512, 512), "red")
        for position, untouched_y, banner_y in [("Top", 500, 30), ("Bottom", 10, 480)]:
            result = pipeline.add_text_overlay(source, "MUMMY & DADDY FOREVER TOGETHER", position=position)
            self.assertEqual(result.getpixel((250, untouched_y)), (255, 0, 0))
            self.assertEqual(result.getpixel((0, banner_y)), (255, 255, 255))
            # No letter pixels may reach the left or right canvas boundaries.
            region = np.asarray(result)[:64] if position == "Top" else np.asarray(result)[-64:]
            self.assertTrue(np.all(region[:, :10] == 255))
            self.assertTrue(np.all(region[:, -10:] == 255))
        self.assertEqual(source.getpixel((0, 0)), (255, 0, 0))


class GenerationTests(unittest.TestCase):
    def test_sdxl_loads_matching_controlnet_and_style_adapter_from_cache(self):
        with patch.object(pipeline, "IS_SDXL", True), patch.object(pipeline, "STYLE_LORA_ID", "local-flat-style"):
            with patch("diffusers.ControlNetModel.from_pretrained") as control, patch("diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained") as base:
                with patch("huggingface_hub.snapshot_download", return_value="cached-style") as snapshot, patch("huggingface_hub.hf_hub_download", return_value="cached-base/model_index.json"):
                    with patch("diffusers.DPMSolverMultistepScheduler.from_config"):
                        result = pipeline.load_diffusion_models()
        self.assertIs(result, base.return_value)
        self.assertNotIn("variant", control.call_args.kwargs)
        self.assertTrue(control.call_args.kwargs["local_files_only"])
        self.assertTrue(base.call_args.kwargs["local_files_only"])
        self.assertEqual(base.call_args.kwargs["variant"], "fp16")
        snapshot.assert_called_once_with("local-flat-style", allow_patterns=[pipeline.STYLE_LORA_FILE], local_files_only=True)
        result.load_lora_weights.assert_called_once_with("cached-style", weight_name=pipeline.STYLE_LORA_FILE, adapter_name="flat-portrait")

    def test_overlong_prompt_fails_before_diffusion_instead_of_silent_truncation(self):
        fake = Mock()
        fake.tokenizer.model_max_length = 77
        fake.tokenizer.return_value = SimpleNamespace(input_ids=list(range(100)))
        with patch.object(pipeline, "get_pipeline", return_value=fake):
            with self.assertRaisesRegex(ValueError, "100 tokens; maximum 77"):
                pipeline.generate(Image.new("RGB", (60, 60)), list(pipeline.STYLE_PROMPTS)[1],
                                  custom_prompt="long instructions", isolate_background=False)
        fake.assert_not_called()

    def test_original_image_edges_strength_and_seed_reach_img2img(self):
        fake = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (512, 512), "blue")]))
        source = Image.new("RGB", (300, 200), "red")
        notices = []
        with patch.object(pipeline, "get_pipeline", return_value=fake):
            edges, output = pipeline.generate(
                source, list(pipeline.STYLE_PROMPTS)[1], custom_prompt="red shirt",
                isolate_background=False, denoise_strength=0.72, control_strength=0.9,
                steps=25, seed=1234, notices=notices,
            )
        params = fake.call_args.kwargs
        self.assertEqual(params["strength"], 0.72)
        self.assertEqual(params["controlnet_conditioning_scale"], 0.9)
        self.assertEqual(params["image"].getpixel((256, 256)), (255, 0, 0))
        self.assertIs(params["control_image"], edges)
        self.assertEqual(params["generator"].initial_seed(), 1234)
        self.assertIn("red shirt", params["prompt"])
        self.assertIn("eyes, nose, mouth", params["negative_prompt"])
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(notices, ["Seed: 1234."])

    def test_each_preset_uses_its_own_defaults(self):
        for name, preset in pipeline.STYLE_PROMPTS.items():
            fake = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (512, 512))]))
            with patch.object(pipeline, "get_pipeline", return_value=fake):
                pipeline.generate(Image.new("RGB", (60, 60)), name, isolate_background=False)
            self.assertEqual(fake.call_args.kwargs["strength"], preset["default_denoise"])
            self.assertEqual(fake.call_args.kwargs["controlnet_conditioning_scale"], preset["default_control"])

    def test_invalid_input_fails_before_loading_models(self):
        style = next(iter(pipeline.STYLE_PROMPTS))
        with patch.object(pipeline, "get_pipeline") as load:
            for kwargs in ({"steps": 19}, {"denoise_strength": 0}, {"seed": -2},
                           {"control_strength": float("nan")}, {"text_overlay": "a" * 81}):
                with self.assertRaises(ValueError):
                    pipeline.generate(Image.new("RGB", (60, 60)), style, **kwargs)
            load.assert_not_called()

    def test_cuda_required_before_model_loading(self):
        with patch.object(pipeline, "_PIPELINE", None), patch.object(pipeline.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU not detected"):
                pipeline.warmup()

    def test_request_cannot_start_a_model_download(self):
        with patch.object(pipeline, "_PIPELINE", None), patch.object(pipeline, "load_diffusion_models") as load:
            with self.assertRaisesRegex(RuntimeError, "Pipeline is not ready"):
                pipeline.generate(Image.new("RGB", (60, 60)), next(iter(pipeline.STYLE_PROMPTS)))
            load.assert_not_called()

    def test_real_step_callback_reports_progress_through_completion(self):
        updates, callbacks = [], []

        def inference(**kwargs):
            for index in range(3):
                data = {"latents": "test"}
                self.assertIs(kwargs["callback_on_step_end"](SimpleNamespace(num_timesteps=3), index, 0, data), data)
            return SimpleNamespace(images=[Image.new("RGB", (512, 512))])

        with patch.object(pipeline, "get_pipeline", side_effect=lambda: inference):
            pipeline.generate(
                Image.new("RGB", (60, 60)), next(iter(pipeline.STYLE_PROMPTS)),
                isolate_background=False, progress=lambda value, desc: updates.append(value),
                step_callback=lambda step, total: callbacks.append((step, total)),
            )
        self.assertEqual(callbacks, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(updates[0], 0)
        self.assertEqual(updates[-1], 1)
        self.assertEqual(updates, sorted(updates))

    def test_warmup_initializes_once_with_cpu_offload(self):
        fake, session = Mock(), Mock()
        with patch.object(pipeline, "_PIPELINE", None), patch.object(pipeline, "_REMBG_SESSION", None):
            with patch.object(pipeline, "load_diffusion_models", return_value=fake) as load:
                with patch.object(pipeline, "create_rembg_session", return_value=session) as background:
                    with patch.object(pipeline.torch.cuda, "is_available", return_value=True):
                        self.assertIs(pipeline.warmup(), fake)
                        self.assertIs(pipeline.warmup(), fake)
                        self.assertIs(pipeline.get_pipeline(), fake)
            load.assert_called_once()
            background.assert_called_once_with(local_files_only=True)
            fake.enable_attention_slicing.assert_called_once()
            fake.vae.enable_slicing.assert_called_once()
            fake.enable_vae_slicing.assert_not_called()
            fake.enable_model_cpu_offload.assert_called_once()

    def test_oom_cleans_offload_hooks_and_reports_recovery(self):
        fake = Mock(side_effect=pipeline.torch.cuda.OutOfMemoryError("test"))
        with patch.object(pipeline, "get_pipeline", return_value=fake), patch.object(pipeline, "_PIPELINE", fake):
            with patch.object(pipeline.torch.cuda, "empty_cache") as clear:
                with self.assertRaisesRegex(RuntimeError, "GPU memory exhausted"):
                    pipeline.generate(Image.new("RGB", (60, 60)), next(iter(pipeline.STYLE_PROMPTS)), isolate_background=False)
        clear.assert_called_once()
        fake.maybe_free_model_hooks.assert_called_once()


class UITests(unittest.TestCase):
    def test_startup_warms_before_launch(self):
        import app
        calls = []
        demo = Mock()
        demo.launch.side_effect = lambda **kwargs: calls.append(("launch", kwargs))
        with patch.object(pipeline, "warmup", side_effect=lambda: calls.append(("warmup", {}))):
            with patch.object(app, "build_ui", return_value=demo):
                app.main()
        self.assertEqual([call[0] for call in calls], ["warmup", "launch"])
        self.assertEqual(calls[1][1]["server_port"], 7860)
        self.assertFalse(calls[1][1]["inbrowser"])

    def test_status_and_output_order(self):
        import app
        artwork = Image.new("RGB", (512, 512), "red")
        edges = Image.new("RGB", (512, 512), "black")
        with patch.object(pipeline, "generate", return_value=(edges, artwork)):
            updates = list(app.run_generation(artwork, app.STYLES[0], "", "", 0.85, 0.7, 25, 7.5, False))
        self.assertIn("Generating", updates[0][2])
        self.assertIs(updates[-1][0], artwork)
        self.assertIs(updates[-1][1], edges)
        self.assertIn("Completed", updates[-1][2])

    def test_ui_builds_and_examples_use_real_files(self):
        import app
        demo = app.build_ui()
        self.assertEqual(demo.title, "ThreadOps AI Art Studio (Local GPU)")
        self.assertEqual(len(app.example_rows()), 4)
        for row in app.example_rows():
            self.assertTrue(os.path.isfile(row[0]))
            self.assertEqual((row[4], row[5]), app.on_style_change(row[1]))


if __name__ == "__main__":
    unittest.main()
