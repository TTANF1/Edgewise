"""Offline regressions: python -m unittest discover -s apps/cli/tests -v."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from edgewise_cli.pipeline import analyze_frame, build_blend_map, decontaminate
from edgewise_cli.main import build_parser
from edgewise_core.mask.background import remove_background
from edgewise_core.edge.detect import nearest_interior_indices
from edgewise_core.edge.normal_sampling import boundary_evidence
from edgewise_semantic.protocol import QualityReport
from edgewise_types.params import DecontaminationParams


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3])
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'input.png'
        self.target = Path(self.directory.name) / 'output.png'
        self.params = DecontaminationParams()

    def frame(self, halo=False):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = self.params.wall_rgb
        arr[8:24, 8:24] = (40, 80, 120, 255)
        if halo:
            arr[8, 8:24, :3] = (125, 149, 174)
        arr[28, 4:28] = (200, 30, 80, 255)
        return arr

    def process(self, arr, **kwargs):
        Image.fromarray(arr).save(self.source)
        stats = decontaminate(str(self.source), str(self.target), self.params,
                             skip_jev=True, **kwargs)
        with Image.open(self.target) as image:
            return np.array(image), stats

    def test_clean_edges_and_thin_line_unchanged(self):
        arr = self.frame()
        actual, stats = self.process(arr)
        np.testing.assert_array_equal(actual, arr)
        self.assertEqual(stats['stop_reason'], 'pixels_stable')

    def test_halo_improves_without_changing_thin_line(self):
        arr = self.frame(True)
        actual, _ = self.process(arr)
        self.assertLess(np.abs(actual[8, 10, :3].astype(float)-[40,80,120]).mean(), 5)
        np.testing.assert_array_equal(actual[28], arr[28])

    def test_no_interior_indices_are_safe(self):
        mask = np.zeros((1, 1), dtype=bool)
        iy, ix = nearest_interior_indices(mask)
        self.assertEqual((iy[0,0], ix[0,0]), (0,0))

    def test_fully_transparent_image_unchanged(self):
        arr = np.zeros((1, 1, 4), dtype=np.uint8)
        actual, _ = self.process(arr)
        np.testing.assert_array_equal(actual, arr)

    def test_single_pixel_subject_unchanged(self):
        arr = np.zeros((5, 5, 4), dtype=np.uint8)
        arr[2,2] = (240,30,20,255)
        actual, _ = self.process(arr)
        np.testing.assert_array_equal(actual, arr)

    def test_explicit_background_removal_preserves_existing_alpha(self):
        arr = self.frame()
        arr[12,12,3] = 91
        im, _, _ = analyze_frame(Image.fromarray(arr), replace(self.params, background_tolerance=30), skip_jev=True)
        self.assertEqual(np.array(im)[12,12,3], 91)

    def test_autodetected_wall_used_in_edge_rules(self):
        arr = self.frame(True)
        arr[arr[:,:,3] == 0] = (255,255,255,255)
        _, _, masks = analyze_frame(Image.fromarray(arr), self.params, skip_jev=True)
        self.assertEqual(masks['params'].wall_rgb, (255,255,255))

    def test_semantic_keep_is_not_overridden(self):
        _, candidates, masks = analyze_frame(Image.fromarray(self.frame(True)), self.params, skip_jev=True)
        reviews = {c.edge_rgb: .1 for c in candidates}
        blend = build_blend_map(masks['edge'], candidates, reviews, self.params, masks['boundary_evidence'])
        self.assertEqual(blend, {})

    def test_quality_stop_always_saves_output_and_cleans_temp_files(self):
        class Reviewer:
            def review(self, candidates):
                return {}
            def evaluate_quality(self, path, wall):
                self.path = Path(path)
                assert self.path.exists()
                return QualityReport(.99, False, [])
        reviewer = Reviewer()
        Image.fromarray(self.frame(True)).save(self.source)
        stats = decontaminate(str(self.source), str(self.target), self.params, reviewer=reviewer)
        self.assertEqual(stats['stop_reason'], 'quality_accepted')
        self.assertTrue(self.target.exists())
        self.assertFalse(reviewer.path.exists())

    def test_zero_iterations_rejected_before_processing(self):
        with self.assertRaisesRegex(ValueError, 'iterations'):
            decontaminate('missing.png', str(self.target), self.params, iterations=0)

    def test_background_preparation_runs_once(self):
        arr = self.frame(True)
        arr[:,:,3] = 255
        Image.fromarray(arr).save(self.source)
        with patch('edgewise_cli.pipeline.remove_background', wraps=remove_background) as remove:
            stats = decontaminate(str(self.source), str(self.target),
                                  replace(self.params, background_tolerance=30), skip_jev=True)
        self.assertEqual(remove.call_count, 1)
        self.assertGreater(stats['iterations'], 1)

    def test_cli_accepts_iteration_limit(self):
        args = build_parser().parse_args(['cutout', 'input', 'output', '--skip-jev', '--iterations', '5'])
        self.assertEqual(args.iterations, 5)

    def test_background_score_direction(self):
        arr = self.frame()
        edge = np.zeros((32,32), dtype=bool)
        edge[8,10] = True
        arr[8,10,:3] = self.params.wall_rgb
        evidence = boundary_evidence(arr.astype(float), edge, ~edge)
        self.assertGreater(evidence[(8,10)]['local_bg_score'], .9)


if __name__ == '__main__':
    unittest.main()
