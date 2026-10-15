from io import BytesIO
import os
import base64
import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import app
import gemini_service


class GeminiServiceTests(unittest.TestCase):
    def test_key_selection_and_missing_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "environment-key"}), patch("google.genai.Client") as client:
            gemini_service.get_client("provided-key")
            client.assert_called_with(api_key="provided-key", vertexai=False)
            gemini_service.get_client()
            client.assert_called_with(api_key="environment-key", vertexai=False)
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaises(ValueError):
                gemini_service.get_client()

    def test_vision_sends_jpeg_and_returns_short_prompt(self):
        client = Mock()
        client.models.generate_content.return_value.text = "A seated black collie with a white chest."
        with patch.object(gemini_service, "get_client", return_value=client):
            result = gemini_service.analyze_and_prompt(Image.new("RGBA", (12, 12)), "Embroidery", "key")
        args = client.models.generate_content.call_args.kwargs
        self.assertEqual(args["model"], "gemini-3.8-flash")
        self.assertIn("Embroidery", args["contents"][0])
        part = args["contents"][1].inline_data
        self.assertEqual(part.mime_type, "image/jpeg")
        with Image.open(BytesIO(part.data)) as photo:
            self.assertEqual(photo.mode, "RGB")
            self.assertEqual(photo.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result, "A seated black collie with a white chest.")
        self.assertLess(len(result.split()), 50)

    def test_vision_rejects_missing_photo_and_empty_description(self):
        with self.assertRaises(ValueError):
            gemini_service.analyze_and_prompt(None, "Vector")
        client = Mock()
        client.models.generate_content.return_value.text = None
        with patch.object(gemini_service, "get_client", return_value=client):
            with self.assertRaisesRegex(ValueError, "no description"):
                gemini_service.analyze_and_prompt(Image.new("RGB", (12, 12)), "Vector")

    def test_real_sdk_sends_developer_api_image_request_and_decodes_response(self):
        import httpx
        from google import genai

        encoded = BytesIO()
        Image.new("RGB", (16, 16), "red").save(encoded, format="PNG")
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
                {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(encoded.getvalue()).decode()}}
            ]}}]})

        with genai.Client(api_key="test-key", vertexai=False, http_options={
            "client_args": {"transport": httpx.MockTransport(respond)},
        }) as client, patch.object(gemini_service, "get_client", return_value=client):
            artwork = gemini_service.generate_with_gemini("red collie", "key", Image.new("RGB", (12, 12)))
            self.assertEqual(artwork.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "generativelanguage.googleapis.com")
        self.assertIn("gemini-nano-banana-2.1:generateContent", str(requests[0].url))
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "red collie")
        self.assertIn("inlineData", payload["contents"][0]["parts"][1])
        self.assertEqual(payload["generationConfig"]["responseModalities"], ["IMAGE"])
        self.assertEqual(payload["generationConfig"]["imageConfig"]["aspectRatio"], "1:1")

    def test_cloud_rejects_empty_response(self):
        from google.genai import types

        client = Mock()
        client.models.generate_content.return_value = types.GenerateContentResponse()
        with patch.object(gemini_service, "get_client", return_value=client):
            with self.assertRaisesRegex(ValueError, "no artwork"):
                gemini_service.generate_with_gemini("red collie", "key")


class CloudGenerationTests(unittest.TestCase):
    def generate(self, **kwargs):
        params = dict(image=Image.new("RGB", (512, 512)), style=app.STYLES[0],
                      custom_prompt="seated collie", text_overlay="OREO", control_scale=0.85,
                      denoise_strength=0.7, steps=25, guidance_scale=7.5, remove_bg=False,
                      api_key="key", engine_choice=app.CLOUD_ENGINE, text_position="Top")
        params.update(kwargs)
        return list(app.run_generation(**params))

    def test_cloud_uses_style_prompt_and_overlay_without_local_generation(self):
        artwork = Image.new("RGB", (512, 512), "red")
        with patch.object(gemini_service, "generate_with_gemini", return_value=artwork) as cloud, patch.object(app.pipeline, "generate") as local:
            output = self.generate()[-1]
        self.assertIn("seated collie", cloud.call_args.kwargs["prompt"])
        self.assertIn(app.pipeline.STYLE_PROMPTS[app.STYLES[0]]["prompt"], cloud.call_args.kwargs["prompt"])
        self.assertIsInstance(cloud.call_args.kwargs["image"], Image.Image)
        local.assert_not_called()
        self.assertEqual(output[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(output[0].getpixel((0, 500)), (255, 0, 0))
        self.assertIsNone(output[1])
        self.assertIn("Gemini Nano Banana 2.1", output[2])

    def test_empty_key_and_hybrid_keep_local_generation(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch.object(app.pipeline, "generate", return_value=(None, Image.new("RGB", (20, 20)))) as local, patch.object(gemini_service, "generate_with_gemini") as cloud:
            self.generate(api_key="")
            self.generate(engine_choice="Hybrid (Gemini Vision + Local GPU)")
        self.assertEqual(local.call_count, 2)
        cloud.assert_not_called()

    def test_cloud_failure_uses_existing_error_status(self):
        with patch.object(gemini_service, "generate_with_gemini", side_effect=RuntimeError("API unavailable")), patch.object(app.gr, "Warning") as warning:
            output = self.generate()[-1]
        warning.assert_called_once_with("API unavailable")
        self.assertEqual(output[:2], (None, None))
        self.assertIn("Generation failed", output[2])
