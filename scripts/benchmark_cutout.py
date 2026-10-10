"""Reproducible, offline cutout benchmark with composites and known-alpha fixtures.

Run: .venv/Scripts/python.exe scripts/benchmark_cutout.py --label baseline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from edgewise_cli.pipeline import decontaminate
from edgewise_types.params import DecontaminationParams

ROOT = Path(__file__).resolve().parents[1]


def fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    wall = (210, 218, 228)
    clean = np.zeros((64, 64, 4), dtype=np.uint8)
    clean[:, :, :3] = wall
    clean[16:48, 16:48] = (40, 80, 120, 255)
    clean[6:10, 5:9] = (220, 80, 30, 255)
    clean[54, 8:55] = (200, 30, 80, 255)
    Image.fromarray(clean).save(directory / 'clean_details.png')
    halo = clean.copy()
    ring = np.zeros((64, 64), dtype=bool)
    ring[16:48, 16:48] = True
    ring[17:47, 17:47] = False
    halo[ring, :3] = (np.array((40, 80, 120)) * .5 + np.array(wall) * .5).astype(np.uint8)
    Image.fromarray(halo).save(directory / 'wall_halo.png')
    return clean, ring


def composite(im, style):
    im = im.convert('RGBA')
    h, w = im.height, im.width
    yy, xx = np.indices((h, w))
    if style == 'checker':
        gray = np.where((xx // 12 + yy // 12) % 2, 180, 230).astype(np.uint8)
        bg = Image.fromarray(np.dstack([gray, gray, gray, np.full_like(gray, 255)]))
    else:
        bg = Image.new('RGBA', im.size, style)
    return Image.alpha_composite(bg, im).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='current')
    args = parser.parse_args()
    root = ROOT / 'artifacts' / 'cutout-review'
    out = root / args.label
    out.mkdir(parents=True, exist_ok=True)
    clean, ring = fixtures(root / 'fixtures')
    cases = [(p.stem, p, DecontaminationParams(background_tolerance=30))
             for p in sorted((ROOT / 'examples/desk8/assets/frames_src').glob('f*.png'))]
    cases += [('favicon', ROOT / 'examples/favicon/input.png', DecontaminationParams())]
    cases += [(name, root / 'fixtures' / f'{name}.png', DecontaminationParams())
              for name in ('clean_details', 'wall_halo')]
    results = []
    preview_cases = []
    for name, source, params in cases:
        start = time.perf_counter()
        target = out / f'{name}.png'
        stats = decontaminate(str(source), str(target), params, skip_jev=True)
        src = np.array(Image.open(source).convert('RGBA'))
        result = np.array(Image.open(target).convert('RGBA'))
        row = dict(name=name, seconds=round(time.perf_counter()-start, 3), **stats)
        row.update(visible_px=int((result[:, :, 3] > 0).sum()),
                   opaque_px=int((result[:, :, 3] == 255).sum()),
                   partial_px=int(((result[:, :, 3] > 0) & (result[:, :, 3] < 255)).sum()))
        if name in ('clean_details', 'wall_halo'):
            fg = clean[:, :, 3] > 0
            row['foreground_alpha_mae'] = float(np.abs(result[:, :, 3].astype(float)-clean[:, :, 3])[fg].mean())
            row['foreground_rgb_mae'] = float(np.abs(result[:, :, :3].astype(float)-clean[:, :, :3])[fg].mean())
            row['thin_line_unchanged'] = bool(np.array_equal(result[54, 8:55], clean[54, 8:55]))
            if name == 'wall_halo':
                row['halo_rgb_mae'] = float(np.abs(result[:, :, :3].astype(float)-clean[:, :, :3])[ring].mean())
        results.append(row)
        if name in ('f01', 'favicon', 'clean_details', 'wall_halo'):
            preview_cases.append((name, source, target))
    # Every case has before/after on white, black and checker backgrounds.
    sheet = Image.new('RGB', (1440, 280 * len(preview_cases)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (name, source, target) in enumerate(preview_cases):
        for col, (path, style) in enumerate((p, s) for s in ('white','black','checker') for p in (source,target)):
            tile = composite(Image.open(path), style)
            if name == 'favicon':
                tile = tile.crop((330, 280, 670, 740))
            scale = min(230/tile.width, 240/tile.height)
            tile = tile.resize((max(1,round(tile.width*scale)), max(1,round(tile.height*scale))),
                               Image.Resampling.NEAREST)
            x, y = col * 240, row * 280
            sheet.paste(tile, (x + (240-tile.width)//2, y+28))
            draw.text((x+6,y+6), f'{name}: {"before" if path == source else "after"} {style}', fill='black')
    sheet.save(out / 'comparison.png')
    if (root / 'baseline/f01.png').exists():
        crops = [('hair', (70,10,175,105)), ('plant', (0,65,65,160)),
                 ('cup_steam', (260,95,310,155)), ('chair', (45,155,215,245))]
        details = Image.new('RGB', (960, 250*len(crops)), 'white')
        pen = ImageDraw.Draw(details)
        for row, (name, box) in enumerate(crops):
            paths = [ROOT/'examples/desk8/assets/frames_src/f01.png',
                     root/'baseline/f01.png', out/'f01.png']
            for col, (path, label) in enumerate(zip(paths, ('source','baseline',args.label))):
                with Image.open(path) as im:
                    tile = composite(im.crop(box), 'checker')
                scale = min(310/tile.width,220/tile.height)
                tile = tile.resize((round(tile.width*scale),round(tile.height*scale)), Image.Resampling.NEAREST)
                details.paste(tile, (col*320, row*250+25))
                pen.text((col*320+6,row*250+6), f'{name}: {label}', fill='black')
        details.save(out/'details.png')
    (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
