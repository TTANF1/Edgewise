"""Jev adapter: TypeSafe System One as a SemanticReviewer."""
from __future__ import annotations

import os
from typing import Sequence

from edgewise_semantic.jev.prompt import build_questions, build_wall_questions
from edgewise_semantic.protocol import QualityReport
from edgewise_types.candidate import EdgeCandidate, RGB

# The SDK is imported lazily inside _get_client() so that the package
# imports cleanly without it (it is an optional dependency).


class JevReviewer:
    """Reviews candidates through the Jev model (TypeSafe System One).

    The API key is read from the ``TYPESAFE_API_KEY`` environment variable
    only. Never embed keys in source — this project is open source.
    """

    def __init__(
        self,
        wall_rgb: RGB = (210, 218, 228),
        scene_hint: str = "developer at desk",
        api_key: str | None = None,
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
    ) -> None:
        self._wall_rgb = wall_rgb
        self._scene_hint = scene_hint
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self._endpoint = endpoint
        self._client = None

    def review(self, candidates: Sequence[EdgeCandidate]) -> dict[RGB, float]:
        if not self._api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Export it before running, or pass "
                "--skip-jev to run the rules-only pipeline."
            )

        # Review ALL candidates (including rule-force) so Jev can downgrade
        # false-positive force pixels that are actually subject detail.
        # Deduplicate by color, keeping the sample with the largest color
        # distance (the most informative example of that color).
        best: dict[RGB, EdgeCandidate] = {}
        for c in candidates:
            prev = best.get(c.edge_rgb)
            if prev is None or c.color_dist > prev.color_dist:
                best[c.edge_rgb] = c

        # Cap at 50 colors to stay within API token limits.
        # Prioritize: non-force uncertain colors first, then force colors.
        sorted_colors = sorted(
            best.items(),
            key=lambda kv: (kv[1].force, -kv[1].color_dist),  # False (non-force) first
        )
        best = dict(sorted_colors[:50])

        if not best:
            return {}

        client = self._get_client()
        colors = [(rgb, info.interior_rgb) for rgb, info in best.items()]
        state = (
            "Pixel art cutout edge analysis. Wall is light grayish-blue. "
            "Checking if edge pixels are contaminated by wall bleed."
        )
        resp = client.system_one(
            state=state,
            questions=build_questions(colors, self._wall_rgb, self._scene_hint),
        )

        result: dict[RGB, float] = {}
        for i, rgb in enumerate(best):
            result[rgb] = float(resp.answers[f"c{i}"].noul)
        return result

    def _get_client(self):
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "typesafe-sdk is not installed. Install the semantic extra: "
                    "pip install -e ./packages/semantic[api]"
                ) from exc
            os.environ["TYPESAFE_API_KEY"] = self._api_key
            self._client = TypeSafeClient()
        return self._client

    def confirm_background_color(
        self,
        candidates: list[RGB],
        interior_sample: RGB,
    ) -> RGB | None:
        """Ask Jev which corner sample is the real wall color."""
        if not self._api_key:
            return None
        if len(candidates) < 2:
            return candidates[0] if candidates else None

        client = self._get_client()
        resp = client.system_one(
            state="Pixel-art scene wall color identification.",
            questions=build_wall_questions(candidates, interior_sample, self._scene_hint),
        )

        best_rgb, best_p = None, 0.0
        for i, rgb in enumerate(candidates):
            p = float(resp.answers[f"w{i}"].noul)
            if p > best_p:
                best_rgb, best_p = rgb, p

        return best_rgb if best_p > 0.5 else None

    def evaluate_quality(
        self,
        frame_path: str,
        wall_rgb: RGB,
    ) -> QualityReport:
        """Judge whether a decontaminated frame still has visible halo."""
        import numpy as np
        from PIL import Image

        img = np.asarray(Image.open(frame_path).convert("RGBA")).astype(float)
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]

        # Find the most suspicious edge pixels: semi-transparent AND bright
        # (bright semi-transparent pixels are classic halo residue)
        semi = (alpha > 30) & (alpha < 200)
        if not semi.any():
            return QualityReport(quality_score=0.95, needs_more_iterations=False, issues=[])

        suspicious = rgb[semi]
        wall = np.array(wall_rgb)
        dist_to_wall = np.sqrt(((suspicious - wall) ** 2).sum(axis=1))

        # Top 5 brightest semi-transparent pixels closest to wall color
        brightness = suspicious.mean(axis=1)
        score = brightness * (1.0 / (dist_to_wall + 10))
        top_idx = np.argsort(-score)[:5]

        candidates: list[tuple[RGB, float]] = []
        for idx in top_idx:
            px = tuple(int(v) for v in suspicious[idx])
            candidates.append((px, float(dist_to_wall[idx])))

        if not candidates:
            return QualityReport(quality_score=0.9, needs_more_iterations=False, issues=[])

        # Ask Jev: are these suspicious pixels still wall-bleed?
        questions: dict[str, dict] = {}
        for i, (px, dist) in enumerate(candidates):
            questions[f"q{i}"] = {
                "type": "noul",
                "instructions": (
                    f"Pixel art cutout quality check. Wall color is RGB{wall_rgb}. "
                    f"Semi-transparent edge pixel RGB({px[0]},{px[1]},{px[2]}) "
                    f"is {dist:.0f} away from wall color. "
                    f"Is this pixel still contaminated by wall bleed (visible halo) "
                    f"that needs another decontamination round?"
                ),
            }

        client = self._get_client()
        resp = client.system_one(
            state="Pixel art cutout quality evaluation.",
            questions=questions,
        )

        halo_count = 0
        issues: list[str] = []
        for i, (px, dist) in enumerate(candidates):
            p = float(resp.answers[f"q{i}"].noul)
            if p > 0.5:
                halo_count += 1
                issues.append(f"halo residue at RGB{px}")

        quality = 1.0 - (halo_count / len(candidates))
        needs_more = halo_count >= 2

        return QualityReport(
            quality_score=quality,
            needs_more_iterations=needs_more,
            issues=issues,
        )
