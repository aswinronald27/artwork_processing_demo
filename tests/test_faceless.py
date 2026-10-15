import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import faceless


class FaceCleanupTests(unittest.TestCase):
    def test_features_removed_without_repainting_hair_clothes_or_source(self):
        image = Image.new("RGB", (300, 300), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((90, 80, 210, 220), fill=(190, 145, 120))
        draw.rectangle((90, 80, 210, 110), fill="black")
        draw.rectangle((90, 221, 210, 299), fill="blue")
        draw.rectangle((120, 135, 127, 139), fill="black")
        draw.rectangle((170, 135, 177, 139), fill="black")
        draw.rectangle((140, 175, 155, 178), fill="black")
        bounds = np.array([[90, 80], [210, 80], [210, 220], [90, 220]], np.float32)
        points = np.array([[123, 137], [173, 137], [150, 155], [143, 176], [154, 176]], np.float32)
        with patch.object(faceless, "detect_faces", return_value=[(bounds, points)]):
            result = faceless.simplify_faces(image)
        self.assertEqual(result.getpixel((123, 137)), (190, 145, 120))
        self.assertEqual(result.getpixel((150, 176)), (190, 145, 120))
        self.assertEqual(result.getpixel((150, 90)), (0, 0, 0))
        self.assertEqual(result.getpixel((150, 250)), (0, 0, 255))
        self.assertEqual(image.getpixel((123, 137)), (0, 0, 0))

    def test_detection_failure_keeps_artwork_and_reports_notice(self):
        image = Image.new("RGB", (50, 50), "red")
        notices = []
        with patch.object(faceless, "detect_faces", side_effect=RuntimeError("weights unavailable")):
            self.assertIs(faceless.simplify_faces(image, notices), image)
        self.assertIn("Face cleanup unavailable", notices[0])

    def test_face_cleanup_supports_dark_skin_tones(self):
        image = Image.new("RGB", (120, 120), (65, 45, 35))
        ImageDraw.Draw(image).rectangle((40, 45, 46, 48), fill="black")
        bounds = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], np.float32)
        points = np.array([[43, 47], [75, 47], [60, 60], [50, 78], [70, 78]], np.float32)
        with patch.object(faceless, "detect_faces", return_value=[(bounds, points)]):
            result = faceless.simplify_faces(image)
        self.assertEqual(result.getpixel((43, 47)), (65, 45, 35))

    def test_offline_missing_face_model_cannot_start_a_download(self):
        with patch.object(faceless.Path, "is_file", return_value=False), patch.object(faceless, "urlretrieve") as download:
            with self.assertRaises(FileNotFoundError):
                faceless.preload_face_model(local_files_only=True)
        download.assert_not_called()
