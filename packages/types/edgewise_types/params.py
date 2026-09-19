"""Tunable parameters for the edge decontamination pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from edgewise_types.candidate import RGB


@dataclass(frozen=True, slots=True)
class DecontaminationParams:
    """All knobs for cutout decontamination, with the desk8 defaults.

    Every value here is a deterministic Layer 1 parameter; the semantic
    layer only contributes a probability that is blended in below.
    """

    # -- geometry --------------------------------------------------------
    alpha_threshold: float = 128.0   # below this, a pixel counts as transparent
    min_color_dist: float = 10.0     # edge/interior pairs closer than this are ignored

    # -- background removal (pre-step, only for opaque wall-backed frames)
    #    None = input is already transparent; a value = flood-fill tolerance.
    background_tolerance: float | None = None

    # -- rule 1: edge much lighter than interior (wall bleeds in) --------
    bright_diff_threshold: float = 20.0
    rule1_sat_delta: float = 0.08    # force when edge is desaturated by this much...
    rule1_sat_min: float = 0.12      # ...and the interior is saturated enough

    # -- rule 2: edge desaturated vs interior (washed out) ---------------
    rule2_sat_delta: float = 0.12
    rule2_sat_min: float = 0.10
    rule2_force_bright_diff: float = 8.0

    # -- rule 3: linear wall-contamination model -------------------------
    wall_rgb: RGB = (210, 218, 228)
    wall_pct_reason: float = 0.25    # report contamination above this
    wall_pct_force: float = 0.40     # force a verdict above this

    # -- blending ----------------------------------------------------------
    base_blend: float = 0.30           # every edge pixel pulls 30% toward interior
    force_blend: float = 0.80          # rule-confirmed pixels pull 80%
    jev_threshold: float = 0.40        # model probability above this -> REMOVE
    jev_uncertain_floor: float = 0.30  # below this -> KEEP
    jev_max_extra_blend: float = 0.50  # max extra pull granted by semantic review
    alpha_decay: float = 0.25          # alpha loss = blend * alpha_decay
