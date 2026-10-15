"""CPU-only face simplification for illustrated portraits; never downloads on requests."""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / ".models/face/face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def preload_face_model(local_files_only: bool = False) -> None:
    if MODEL_PATH.is_file():
        return
    if local_files_only:
        raise FileNotFoundError("Face cleanup weights are missing. Run preload_models.py while online first.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".download")
    try:
        urlretrieve(MODEL_URL, temporary)
        if temporary.stat().st_size < 200_000:
            raise ValueError("Downloaded face detector is incomplete.")
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def detect_faces(image: Image.Image) -> list[tuple[np.ndarray, np.ndarray]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Face cleanup weights are missing; run the portrait preloader.")
    pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = pixels.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), score_threshold=0.5)
    result = []
    for angle in (0, 90, 180, 270):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated = cv2.warpAffine(pixels, matrix, (width, height), borderValue=(255, 255, 255))
        _, faces = detector.detect(rotated)
        if faces is None:
            continue
        inverse = cv2.invertAffineTransform(matrix)
        for face in faces:
            points = cv2.transform(face[4:14].reshape(1, 5, 2), inverse)[0]
            if any(np.linalg.norm(points.mean(axis=0) - old.mean(axis=0)) < max(20, width * 0.07)
                   for _, old in result):
                continue
            x, y, w, h = face[:4]
            corners = np.array([[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]], np.float32)
            bounds = cv2.transform(corners, inverse)[0]
            result.append((bounds, points))
    return result


def simplify_faces(image: Image.Image, notices: list[str] | None = None) -> Image.Image:
    """Flatten face skin while preserving dark hair and regions outside detected faces."""
    try:
        faces = detect_faces(image)
    except Exception as exc:
        if notices is not None:
            notices.append(f"Face cleanup unavailable: {exc}")
        return image
    pixels = np.asarray(image.convert("RGB")).copy()
    for bounds, points in faces:
        region = np.zeros(pixels.shape[:2], np.uint8)
        cv2.fillConvexPoly(region, bounds.astype(np.int32), 255)
        features = np.zeros_like(region)
        cv2.fillConvexPoly(features, cv2.convexHull(points.astype(np.int32)), 255)
        radius = max(5, round(float(np.ptp(points, axis=0).max()) * 0.6))
        features = cv2.dilate(features, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
        rgb = pixels.astype(np.float32)
        skin = (rgb[:,:,0] > rgb[:,:,1]*0.98) & (rgb[:,:,1] > rgb[:,:,2]*0.98) & (rgb[:,:,0] > 35) & (rgb[:,:,0]-rgb[:,:,2] > 6)
        samples = pixels[(features > 0) & (region > 0) & skin]
        if len(samples) < 10:
            continue
        tone = np.median(samples, axis=0).astype(np.uint8)
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).astype(np.float32)
        tone_lab = cv2.cvtColor(tone.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        matches = skin & (np.linalg.norm(lab[:,:,1:] - tone_lab[1:], axis=2) < 10) & (np.abs(lab[:,:,0] - tone_lab[0]) < 25) & (np.linalg.norm(rgb-tone.astype(np.float32),axis=2) < 40)
        mask = ((region > 0) & matches).astype(np.uint8)*255
        kernel = max(3, radius // 2) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
        mask = cv2.bitwise_and(mask, region)
        alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[:,:,None] / 255
        alpha *= (region > 0)[:,:,None]
        pixels = (pixels*(1-alpha) + tone*alpha).astype(np.uint8)
    if notices is not None:
        notices.append(f"Face cleanup: {len(faces)} detected. Inspect for missed features or unwanted changes.")
    return Image.fromarray(pixels)
