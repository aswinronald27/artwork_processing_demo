"""Read JPGs from an explicit path list and make local artwork review sheets.

White-background detection only prioritizes candidates; selection is visual.
No source file is changed or uploaded.
"""
from pathlib import Path
import argparse
import json
import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "work-artwork-audit"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="outputs/work-jpg-paths.txt")
    parser.add_argument("--output", default="outputs/work-artwork-audit")
    args = parser.parse_args()
    OUT = ROOT / args.output
    OUT.mkdir(parents=True, exist_ok=True)
    path_file = ROOT / args.paths
    encoding = "utf-16" if path_file.read_bytes().startswith(b"\xff\xfe") else "utf-8-sig"
    paths = sorted(set(path_file.read_text(encoding=encoding).splitlines()))
    records, unique, candidates = [], {}, []
    for index, name in enumerate(paths):
        path = Path(name)
        row = {"id": index + 1, "path": name}
        try:
            with Image.open(path) as raw:
                row["size"] = list(raw.size)
                im = ImageOps.exif_transpose(raw).convert("RGB")
                im.thumbnail((256, 256))
                digest = hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()
                row["thumbnail_hash"] = digest
                pixels = np.asarray(im)
                white = np.all(pixels > 235, axis=2)
                border = np.concatenate([white[:4].ravel(), white[-4:].ravel(), white[:, :4].ravel(), white[:, -4:].ravel()])
                row.update(white_fraction=float(white.mean()), white_border=float(border.mean()))
                if digest in unique:
                    row["duplicate_of"] = unique[digest]
                else:
                    unique[digest] = row["id"]
                    if min(raw.size) >= 256 and white.mean() > .18 and border.mean() > .45:
                        row["candidate"] = True
                        candidates.append(row)
                        im.save(OUT / f"thumb-{row['id']:04d}.jpg", quality=90)
        except Exception as exc:
            row["error"] = str(exc)
        records.append(row)
        if (index + 1) % 100 == 0:
            print(f"Read {index + 1}/{len(paths)}; candidates {len(candidates)}", flush=True)
    (OUT / "inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for page in range(math.ceil(len(candidates) / 96)):
        selected = candidates[page*96:(page+1)*96]
        board = Image.new("RGB", (1536, 1200), "#cccccc")
        draw = ImageDraw.Draw(board)
        for index, row in enumerate(selected):
            with Image.open(OUT / f"thumb-{row['id']:04d}.jpg") as im:
                thumb = ImageOps.contain(im, (124, 128))
            x, y = (index % 12) * 128, (index // 12) * 150
            board.paste(thumb, (x+(128-thumb.width)//2, y+20+(128-thumb.height)//2))
            draw.text((x+4, y+4), str(row["id"]), fill="black")
        board.save(OUT / f"candidates-{page+1:02d}.jpg", quality=94)
    print(f"Done: {len(records)} image paths; {len(candidates)} unique white-background candidates; {sum('error' in r for r in records)} read errors.")


if __name__ == "__main__":
    main()
