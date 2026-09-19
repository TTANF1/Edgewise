# desk8 — pixel-art cutout decontamination example

The first end-to-end project that shaped Edgewise: an 8-frame
developer-at-desk animation exported from Aseprite as opaque RGB PNGs
on a light grayish-blue wall.

## Assets (self-contained in this directory)

| Path | Role |
| --- | --- |
| `assets/frames_src/f01..f08.png` | **Before** — opaque RGB frames exported from the Aseprite project, wall background included |
| `assets/frames_clean/f01..f08.png` | **After** — wall removed + edge decontaminated, transparent |
| `assets/before.png` | Sprite sheet built from `frames_src` (wall visible) |
| `assets/after.png` | Sprite sheet built from `frames_clean` (wall removed) |
| `desk8.json` | Config with paths relative to this directory |

`before.png` and `after.png` are the comparison pair: put them side by
side to see the wall removal and edge tightening on every frame.

The example is fully self-contained — it does not reference any path
outside the repository. The Aseprite project file lives at
`D:/Aseprite-v1.3.18-Source/animation/desk8/desk-animation8.aseprite`.

## Workflow

```bash
# 1. Export frames from the Aseprite project (opaque RGB, wall included)
aseprite -b desk-animation8.aseprite --save-as "f{frame}.png"

# 2. Cutout: remove wall + decontaminate edges
edgewise cutout --config desk8.json

#    Offline alternative — deterministic rules only, no API calls:
edgewise cutout --config desk8.json --skip-jev

# 3. Rebuild sprite sheet + animated WebPs (sheet saved as after.png)
edgewise rebuild --config desk8.json

#    Regenerate the before/after comparison sheets only:
edgewise rebuild assets/frames_src   assets --sheet-name before.png --sheet-only
edgewise rebuild assets/frames_clean assets --sheet-name after.png  --sheet-only
```

Config paths are resolved relative to `desk8.json`, so the commands above
work from any working directory. Running them regenerates
`assets/frames_clean` and `assets/after.png` from the current tool; the
committed copies are the reference baseline.

## Pipeline

```
input opaque RGB frame (wall background)
  ├─ 0. Background removal   flood-fill wall color from borders (Layer 1)
  ├─ 1. Edge detection       opaque pixels adjacent to transparency
  ├─ 2. Neighbor analysis    nearest interior pixel (distance transform)
  ├─ 3. Rule marking         Rule 1/2/3 -> force flag (no API)
  ├─ 4. Jev semantic review  ambiguous colors only, one noul question each
  └─ 5. Color decontamination RGB pulled toward interior, alpha tightened
```

Step 0 only runs when `background_tolerance` is set (see config). Without
it, the tool assumes the input is already transparent and starts at step 1.

## Default parameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| `WALL_RGB` | (210, 218, 228) | wall reference color |
| `background_tolerance` | 30 | flood-fill tolerance for wall removal |
| base blend | 0.30 | every edge pixel pulls 30% toward interior |
| force blend | 0.80 | rule-confirmed pixels pull 80% |
| Jev threshold | 0.4 | probability > 0.4 -> REMOVE |
| bright diff | 20 | edge brighter than interior by this much is suspicious |
| saturation diff | 0.12 | edge desaturated by this much is suspicious |
| wall model | 40% | reconstructed contamination above 40% forces REMOVE |
| alpha decay | 0.25 | alpha loss = blend × 0.25 |

## Tuning guide

- **Wall still visible around the character**: raise `background_tolerance` (30 → 40).
- **Subject edges eroded / detail eaten**: lower `background_tolerance` (30 → 20).
- **Halo still visible on edges**: raise base blend (0.30 → 0.35) or force blend (0.80 → 0.90).
- **Edges too thin / chunks missing**: lower base blend (0.30 → 0.20).
- **Subject detail wrongly removed**: raise the Jev threshold (0.4 → 0.5).
- **Different background color**: change `wall_rgb` to the new background.

All knobs live in `DecontaminationParams` (`packages/types`); the JSON config
can override any field by name.
