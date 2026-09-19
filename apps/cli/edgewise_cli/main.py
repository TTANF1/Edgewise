"""edgewise command-line interface."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# Load .env from the project root (or cwd) before reading env vars.
from dotenv import load_dotenv

load_dotenv()  # looks for .env in cwd and parents; no-op if missing

from edgewise_types.params import DecontaminationParams

from edgewise_cli.pipeline import decontaminate
from edgewise_cli.rebuild import rebuild_products


# ---------------------------------------------------------------- helpers

def _positive_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _rgb(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
        raise argparse.ArgumentTypeError("expected R,G,B with 0-255 values")
    return (parts[0], parts[1], parts[2])


def load_config(path: str | None) -> tuple[dict[str, Any], str | None]:
    """Read a config JSON. Expected shape:

    {
      "cutout":  {"src_dir": ..., "out_dir": ..., "pattern": "f*.png",
                  "params": {..., "scene_hint": "developer at desk"}},
      "rebuild": {"frames_dir": ..., "out_dir": ..., "pattern": "f*.png",
                  "duration_ms": 165, "webp_quality": 90, "sheet_name": ...}
    }

    Returns (data, base_dir). Paths inside the config are resolved relative
    to the config file's directory, so examples are self-contained.
    """
    if not path:
        return {}, None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), os.path.dirname(os.path.abspath(path))


def _resolve_config_path(value: Any, base: str | None) -> str | None:
    """Resolve a config path relative to the config file's directory."""
    if not value or not base or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base, value))


def _pick_path(value_cli: str | None, value_cfg: Any, base: str | None) -> str | None:
    """CLI positional args win; config values resolve relative to the config."""
    if value_cli:
        return value_cli
    return _resolve_config_path(value_cfg, base)


def merge_params(
    params: DecontaminationParams,
    cfg_params: dict[str, Any] | None,
    args: argparse.Namespace,
) -> DecontaminationParams:
    """params = defaults < config file < command-line flags."""
    overrides: dict[str, Any] = {}
    for key, value in (cfg_params or {}).items():
        if hasattr(params, key):
            overrides[key] = tuple(value) if key == "wall_rgb" else value
    for key in ("wall_rgb", "jev_threshold", "base_blend", "force_blend"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "remove_background", None) is not None:
        overrides["background_tolerance"] = args.remove_background
    return replace(params, **overrides)


# ---------------------------------------------------------------- commands

def cmd_cutout(args: argparse.Namespace) -> int:
    cfg, base = load_config(args.config)
    cut = cfg.get("cutout", {})

    src_dir = _pick_path(args.src_dir, cut.get("src_dir"), base)
    if not src_dir:
        print("error: src_dir is required (positional argument or config)", file=sys.stderr)
        return 2
    if not os.path.isdir(src_dir):
        print(f"error: src_dir not found: {src_dir}", file=sys.stderr)
        return 2
    out_dir = _pick_path(args.out_dir, cut.get("out_dir"), base)
    if not out_dir:
        print("error: out_dir is required (positional argument or config)", file=sys.stderr)
        return 2
    pattern = args.pattern or cut.get("pattern", "f*.png")

    params = merge_params(DecontaminationParams(), cut.get("params"), args)

    paths = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not paths:
        print(f"error: no frames matched '{pattern}' in {src_dir}", file=sys.stderr)
        return 2

    reviewer = None
    if not args.skip_jev:
        from edgewise_semantic.jev.adapter import JevReviewer

        scene_hint = (cut.get("params") or {}).get("scene_hint", "developer at desk")
        reviewer = JevReviewer(wall_rgb=params.wall_rgb, scene_hint=scene_hint)

    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        name = os.path.basename(path)
        print(f"Processing {name}...")
        decontaminate(
            path,
            os.path.join(out_dir, name),
            params,
            reviewer=reviewer,
            skip_jev=args.skip_jev,
        )
    print(f"done -> {out_dir}")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    cfg, base = load_config(args.config)
    reb = cfg.get("rebuild", {})

    frames_dir = _pick_path(args.frames_dir, reb.get("frames_dir"), base)
    if not frames_dir:
        print("error: frames_dir is required (positional argument or config)", file=sys.stderr)
        return 2
    if not os.path.isdir(frames_dir):
        print(f"error: frames_dir not found: {frames_dir}", file=sys.stderr)
        return 2
    out_dir = _pick_path(args.out_dir, reb.get("out_dir"), base)
    if not out_dir:
        print("error: out_dir is required (positional argument or config)", file=sys.stderr)
        return 2

    rebuild_products(
        frames_dir=frames_dir,
        out_dir=out_dir,
        pattern=args.pattern or reb.get("pattern", "f*.png"),
        duration_ms=args.duration or reb.get("duration_ms", 165),
        webp_quality=args.webp_quality or reb.get("webp_quality", 90),
        lossless=not args.no_lossless,
        sheet_name=args.sheet_name or reb.get("sheet_name", "sprite-sheet.png"),
        sheet_only=args.sheet_only,
    )
    print(f"done -> {out_dir}")
    return 0


# ------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgewise",
        description="Pixel-art cutout decontamination: deterministic rules + semantic review.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cut = sub.add_parser("cutout", help="Clean edge bleed from transparent PNG frames")
    cut.add_argument("src_dir", nargs="?", help="Directory with transparent frames")
    cut.add_argument("out_dir", nargs="?", help="Output directory for cleaned frames")
    cut.add_argument("--config", help="JSON config file (paths + params)")
    cut.add_argument("--pattern", help="Frame glob pattern (default f*.png)")
    cut.add_argument("--wall-rgb", type=_rgb, help="Wall reference color R,G,B")
    cut.add_argument("--jev-threshold", type=_positive_float, help="Semantic REMOVE threshold (default 0.4)")
    cut.add_argument("--base-blend", type=_positive_float, help="Base edge pull toward interior (default 0.30)")
    cut.add_argument("--force-blend", type=_positive_float, help="Rule-confirmed pull (default 0.80)")
    cut.add_argument("--skip-jev", action="store_true", help="Run rules only; no API calls")
    cut.add_argument(
        "--remove-background",
        nargs="?",
        const=30.0,
        type=float,
        help="Flood-fill wall color to transparent first (for opaque RGB frames); optional tolerance (default 30)",
    )

    reb = sub.add_parser("rebuild", help="Rebuild sprite sheet + animated WebP from cleaned frames")
    reb.add_argument("frames_dir", nargs="?", help="Directory with cleaned frames")
    reb.add_argument("out_dir", nargs="?", help="Output directory")
    reb.add_argument("--config", help="JSON config file")
    reb.add_argument("--pattern", help="Frame glob pattern (default f*.png)")
    reb.add_argument("--duration", type=int, help="Per-frame duration in ms (default 165)")
    reb.add_argument("--webp-quality", type=int, help="WebP quality (default 90)")
    reb.add_argument("--no-lossless", action="store_true", help="Skip the lossless WebP output")
    reb.add_argument("--sheet-name", help="Sprite sheet filename (default sprite-sheet.png)")
    reb.add_argument("--sheet-only", action="store_true", help="Only build the sprite sheet, no WebPs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "cutout":
        return cmd_cutout(args)
    if args.command == "rebuild":
        return cmd_rebuild(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
