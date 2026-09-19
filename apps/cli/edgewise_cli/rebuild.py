"""Rebuild sprite sheet and animated WebP from cleaned frames."""
from __future__ import annotations

import glob
import os
from typing import Any

from PIL import Image


def rebuild_products(
    frames_dir: str,
    out_dir: str,
    pattern: str = "f*.png",
    duration_ms: int = 165,
    webp_quality: int = 90,
    lossless: bool = True,
    sheet_name: str = "sprite-sheet.png",
    sheet_only: bool = False,
) -> dict[str, Any]:
    """Build a sprite sheet (+ animated WebPs unless sheet_only) from frames."""
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No frames matched '{pattern}' in {frames_dir}")
    frames = [Image.open(p).convert("RGBA") for p in paths]

    w, h = frames[0].size
    n = len(frames)
    print(f"{n} frames, {w}x{h} each")

    # Horizontal sprite sheet
    sheet = Image.new("RGBA", (w * n, h))
    for i, fr in enumerate(frames):
        sheet.paste(fr, (i * w, 0))
    sheet_path = os.path.join(out_dir, sheet_name)
    sheet.save(sheet_path)
    print(f"{sheet_name} saved ({sheet_path})")

    if sheet_only:
        return {"frames": n, "size": (w, h), "outputs": {"sheet": sheet_path}}

    outputs: dict[str, str] = {"sheet": sheet_path}

    # Lossless animated WebP
    if lossless:
        lossless_path = os.path.join(out_dir, "anim-lossless.webp")
        frames[0].save(
            lossless_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            lossless=True,
            quality=100,
        )
        outputs["lossless_webp"] = lossless_path
        print(f"anim-lossless.webp saved ({lossless_path})")

    # Quality-90 animated WebP
    webp_path = os.path.join(out_dir, "anim.webp")
    frames[0].save(
        webp_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=webp_quality,
        method=6,
    )
    outputs["webp"] = webp_path
    print(f"anim.webp saved ({webp_path})")

    return {"frames": n, "size": (w, h), "outputs": outputs}
