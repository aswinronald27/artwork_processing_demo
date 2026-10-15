"""Prepare the visually audited ThreadOps examples without altering originals.

This prepares a conservative flat pet pilot, not a trained model. Other styles
remain listed in the manifest for independent training runs.
"""
from pathlib import Path
import hashlib
import json
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Demo Images"
OUTPUT = ROOT / "training" / "prepared"
PETS = {1, 2, 5, 8, 9, 11, 12, 15, 19, 20, 22, 24, 25, 27, 28, 32,
        38, 39, 40, 41, 42, 43, 45, 46, 47, 49}
MIXED = {6, 14, 26}
ART_FIRST = {1, 8, 21, 50}
# These are visual style/composition flags, not unreadable or damaged files.
REVIEW = {
    8: "Softer shading than the requested crisp flat style",
    9: "Multiple full-body pets; reserve for a separate composition group",
    22: "Detailed soft fur rendering differs from the flat target",
    28: "Black background; needs reviewed background normalization",
    38: "Black background; needs reviewed background normalization",
    41: "Multiple full-body pets; reserve for a separate composition group",
    46: "Multiple full-body pets; reserve for a separate composition group",
    49: "Softer fur rendering than the flat target; retain as a separate cat reference",
}
HOLDOUT = {5, 20, 45}
CAPTIONS = {
    1: "a black tortoiseshell kitten head, orange forehead markings, round amber eyes, white chin",
    2: "a tan and white dog head, upright ears, white forehead stripe, tongue visible",
    5: "a fawn pug head, black muzzle, folded dark ears, blue collar",
    11: "a yellow dog head in side profile, purple collar, open mouth, tongue visible",
    12: "an elderly tan and white dog head, floppy ears, gray muzzle, open mouth",
    15: "a black dog head, floppy ears, closed mouth, three-quarter view",
    19: "a golden dog head, floppy ears, pale muzzle, closed mouth",
    20: "a tan and black shepherd dog head, tall upright ears, closed mouth",
    24: "a golden dog head facing forward, floppy ears, open mouth, pink tongue",
    25: "a black dog head facing forward, floppy ears, open mouth, pink tongue",
    27: "a tan dog head in three-quarter view, one raised ear, patterned clothing",
    32: "a pale yellow dog head in side profile, floppy ears, closed mouth",
    39: "a brown and white boxer dog head, dark muzzle, pink and black nose, collar",
    40: "a tan black and white dog head, floppy ears, orange collar, closed mouth",
    42: "a tan and black shepherd dog head in side profile, upright ears, open mouth",
    43: "a golden dog head, floppy ears, red bow tie and black formal collar",
    45: "a tan dog head, folded ears, dark muzzle, colorful collar, tongue visible",
    47: "a black and white husky dog head facing forward, upright ears, tongue visible",
}


def prepare_artwork(image):
    """Crop white margins so training does not learn tiny subjects on large pages."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    foreground = np.any(np.asarray(image) < 235, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        raise ValueError("Artwork has no visible foreground")
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= max(12, largest * 0.002)) + 1
    ys, xs = np.nonzero(np.isin(labels, keep))
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    crop = ImageOps.contain(crop, (448, 448), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), "white")
    canvas.paste(crop, ((512 - crop.width) // 2, (512 - crop.height) // 2))
    return canvas


def artwork_signature(image):
    """Compare normalized composition plus color, including recompressed JPGs."""
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    colors = np.asarray(image.resize((32, 32)), dtype=np.float32)
    return bits, colors


def is_near_duplicate(first, second):
    return np.count_nonzero(first[0] != second[0]) <= 6 and np.mean(np.abs(first[1] - second[1])) < 12


def add_supplement(records, metadata):
    selection_file = ROOT / "training" / "supplement_selection.json"
    if not selection_file.is_file():
        return
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    signatures = []
    for record in records:
        if record.get("prepared_artwork"):
            with Image.open(ROOT / record["prepared_artwork"]) as im:
                signatures.append((record["pair_id"], artwork_signature(im)))
    copies = ROOT / "training" / "data" / "work-artwork"
    copies.mkdir(parents=True, exist_ok=True)
    skipped = []
    for item in selection["artworks"]:
        source = Path(item["source"])
        local_copy = copies / (item["id"] + source.suffix.lower())
        read_path = local_copy if local_copy.is_file() else source
        with Image.open(read_path) as im:
            prepared = prepare_artwork(im)
        signature = artwork_signature(prepared)
        duplicate = next((name for name, previous in signatures if is_near_duplicate(signature, previous)), None)
        if duplicate is not None:
            skipped.append({"id": item["id"], "duplicate_of": duplicate})
            continue
        if not local_copy.is_file():
            shutil.copy2(source, local_copy)
        target = OUTPUT / "pets" / "train" / (item["id"] + ".png")
        prepared.save(target)
        caption = "threadopsflat, " + item["description"] + ", flat vector illustration, white background"
        metadata.append({"file_name": target.name, "text": caption})
        records.append({
            "pair_id": item["id"], "category": "pets", "split": "train",
            "species": item["species"], "photo": None,
            "artwork": local_copy.relative_to(ROOT).as_posix(),
            "original_source": str(source), "prepared_artwork": target.relative_to(ROOT).as_posix(),
            "caption": caption, "note": "Visually selected standalone artwork; no photo pairing assumed",
        })
        signatures.append((item["id"], signature))
        if len(metadata) >= selection.get("target_training_count", 50):
            break
    (OUTPUT / "supplement-duplicates.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, metadata, seen = [], [], {}
    for number in range(1, 51):
        files = sorted(f for f in (SOURCE / str(number)).iterdir() if f.is_file())
        if len(files) != 2:
            raise ValueError(f"Folder {number} needs exactly two files; found {len(files)}")
        art, photo = (files[0], files[1]) if number in ART_FIRST else (files[1], files[0])
        dimensions = {}
        for role, file in (("artwork", art), ("photo", photo)):
            with Image.open(file) as im:
                im.load()
                dimensions[role] = list(im.size)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate files: {file} and {seen[digest]}")
            seen[digest] = str(file)
        category = "pets" if number in PETS else "mixed" if number in MIXED else "people"
        split = ("review" if number in REVIEW else "holdout" if number in HOLDOUT else "train") if category == "pets" else "separate_style"
        record = {
            "pair_id": str(number), "category": category, "split": split,
            "photo": photo.relative_to(ROOT).as_posix(),
            "artwork": art.relative_to(ROOT).as_posix(), "dimensions": dimensions,
            "note": REVIEW.get(number, ""),
        }
        if category == "pets" and split in ("train", "holdout"):
            folder = OUTPUT / "pets" / split
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"pet_{number:03d}.png"
            with Image.open(art) as im:
                prepare_artwork(im).save(target)
            caption = "threadopsflat, " + CAPTIONS[number] + ", flat vector illustration, white background"
            record.update(prepared_artwork=target.relative_to(ROOT).as_posix(), caption=caption)
            if split == "train":
                metadata.append({"file_name": target.name, "text": caption})
        records.append(record)
    add_supplement(records, metadata)
    (OUTPUT / "pets" / "train" / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8",
    )
    manifest = {
        "status": "prepared_not_trained", "classification": "manually visually reviewed",
        "holdout_identity_note": "Filename separation verified. Same-pet identity across folders needs owner confirmation before final evaluation.",
        "pairs": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(metadata)} flat pet training artworks; {len(HOLDOUT)} pet holdout pairs.")
    print(f"Original paired dataset: pet-only {len(PETS)}; mixed {len(MIXED)}; people {50 - len(PETS) - len(MIXED)}.")
    print(f"Flagged {len(REVIEW)} pet examples for style/background/composition review.")
    print("Original photos/artworks unchanged. No model training has started.")


if __name__ == "__main__":
    main()
