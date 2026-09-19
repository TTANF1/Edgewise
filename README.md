# Edgewise

Deterministic pixel-art cutout processing with a **swappable semantic review layer**.

Edgewise removes the background bleed (halo) left on the edges of transparent PNG
cutouts exported from tools like Aseprite. The pipeline is split into three layers so
that all geometry, color math and morphology are **100% deterministic**, and the only
"judgment call" is delegated to a semantic model that can be replaced without touching
the graphics layer.

## Current status

Working end-to-end. The `examples/desk8` project — an 8-frame developer-at-desk
animation exported from Aseprite — runs `cutout` → `rebuild` and produces cleaned
frames, a horizontal sprite sheet and animated WebPs.

## Architecture

```
Layer 1 — Graphics (deterministic · packages/core)
  RGB/HSL conversion · color distance · luminance · blending
  edge detection · nearest-interior lookup · morphology · masks
  connected components
        │  EdgeCandidate[]   ("who might be contaminated, and why")
        ▼
Layer 2 — Semantic (packages/semantic)
  Candidate → model verdict → KEEP | REMOVE | UNCERTAIN
  (current backend: Jev / TypeSafe System One)
        │  {edge color: contamination probability}
        ▼
Layer 3 — Apps & runtimes
  apps/cli      : Python pipeline — what ships today
  apps/web      : browser demo (WASM / WebGPU) — planned
  packages/wasm : WASM build of Layer 1 — planned
```

Why the split? If Jev turns out to be the wrong model for a given decision, you swap
the adapter in `packages/semantic` and the graphics layer never changes. Layer 1 is
pure numpy/scipy — deterministic, unit-testable, and portable to WASM/WebGPU later.

## Repository layout

```
edgewise/
├── packages/
│   ├── core/                  # Layer 1 — deterministic graphics
│   │   └── edgewise_core/
│   │       ├── pixel/         # color math (HSL, luminance, distance, blending)
│   │       ├── edge/          # edge detection, nearest-interior, contamination rules
│   │       ├── mask/          # morphology / mask ops
│   │       └── segmentation/  # connected components (ready for future use)
│   ├── semantic/              # Layer 2 — swappable semantic backends
│   │   └── edgewise_semantic/
│   │       ├── protocol.py    # SemanticReviewer contract
│   │       └── jev/           # Jev adapter (TypeSafe System One)
│   ├── types/                 # shared contracts (candidates, params, decisions)
│   │   └── edgewise_types/
│   └── wasm/                  # WASM build of Layer 1 — planned
├── apps/
│   ├── cli/                   # cutout + rebuild commands
│   │   └── edgewise_cli/
│   └── web/                   # browser demo — planned
└── examples/
    └── desk8/                 # working Aseprite cutout example
```

## Getting started

Prerequisites: Python 3.10+, and `uv` (recommended) or pip.

```bash
uv sync
edgewise --help
```

Plain pip fallback:

```bash
pip install -e packages/types -e packages/core -e packages/semantic -e apps/cli
```

Run the desk8 example (see `examples/desk8/README.md` for details):

```bash
edgewise cutout   --config examples/desk8/desk8.json
edgewise rebuild  --config examples/desk8/desk8.json
```

Run without installing (from the repo root):

```bash
PYTHONPATH="packages/types;packages/core;packages/semantic;apps/cli" \
  python -m edgewise_cli.main cutout --skip-jev --config examples/desk8/desk8.json
```

## Security

The Jev adapter reads the API key from the `TYPESAFE_API_KEY` environment variable.
Keys are never stored in the repository. If you fork this project and find a key in
older history, treat it as compromised and rotate it.

## Roadmap

- pytest coverage for Layer 1 and the rule engine
- segmentation: region pruning / island removal on top of connected components
- WASM/WebGPU runtime for Layer 1, browser demo in `apps/web`
- additional semantic backends (local heuristics, CLIP-style models) behind the same
  `SemanticReviewer` protocol

## License

MIT © 2026 Yao
