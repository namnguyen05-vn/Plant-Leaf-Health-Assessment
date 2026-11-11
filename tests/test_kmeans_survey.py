"""H4.1 contracts, independent of the application pipeline."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from experiments import kmeans_survey as survey


class SurveyTests(unittest.TestCase):
    def setUp(self):
        self.rgb, self.roi = survey.synthetic_scene()

    def test_twelve_configurations_and_both_initializers(self):
        for init in ('random', 'k-means++'):
            configs = survey.configurations(init)
            self.assertEqual(len(configs), 12)
            self.assertEqual({(c['color_space'], c['k']) for c in configs},
                             {(s, k) for s in ('RGB', 'HSV', 'Lab') for k in range(2, 6)})
            self.assertTrue(all(c['init'] == init for c in configs))
        with self.assertRaises(ValueError):
            survey.configurations('bad')

    def test_rgb_conversion(self):
        result = survey.convert_features(self.rgb, 'RGB')
        np.testing.assert_array_equal(result, self.rgb)
        self.assertFalse(np.shares_memory(result, self.rgb))

    def test_hsv_convention(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 8]]], np.uint8)
        features = survey.convert_features(rgb, 'HSV')
        np.testing.assert_array_equal(features[0, :3], [[0, 255, 255], [85, 255, 255], [171, 255, 255]])
        raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        np.testing.assert_array_equal(features[..., 0], np.rint(raw[..., 0].astype(float)*255/179))
        np.testing.assert_array_equal(features[..., 1:], raw[..., 1:])

    def test_lab_convention(self):
        rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], np.uint8)
        actual = survey.convert_features(rgb, 'Lab')
        np.testing.assert_array_equal(actual, cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab))
        np.testing.assert_array_equal(actual[0, :2], [[0, 128, 128], [255, 128, 128]])

    def test_roi_membership_and_nonmutation(self):
        before = self.roi.copy()
        roi = self.roi.astype(float) * 7
        roi[0, 0] = -2
        np.testing.assert_array_equal(survey.validate_roi(self.rgb, roi), before)
        np.testing.assert_array_equal(self.roi, before)

    def test_inertia_roi_and_returned_centers(self):
        features = np.array([[[1, 2, 3], [4, 5, 6], [255, 255, 255]]], np.uint8)
        centers = np.array([[0, 0, 0], [3, 3, 3]], np.float32)
        labels = np.array([[0, 1, -1]], np.int32)
        self.assertEqual(survey.inertia(features, labels, centers, np.array([[1, 1, 0]])), 28.)

    def test_mask_metrics_partition_and_ratios(self):
        roi = np.array([[1, 1, 1, 0]], bool)
        result = dict(labels_2d=np.array([[0, 1, 2, -1]]))
        for i, name in enumerate(('disease', 'healthy', 'uncertain')):
            mask = np.zeros((1, 4), np.uint8)
            mask[0, i] = 255
            result[name + '_mask'] = mask
        values = survey.mask_metrics(result, roi)
        for name in ('disease', 'healthy', 'uncertain'):
            self.assertEqual(values[name + '_pixel_count'], 1)
            self.assertAlmostEqual(values[name + '_ratio_percent'], 100/3)
        result['healthy_mask'][0, 0] = 255
        with self.assertRaises(ValueError):
            survey.mask_metrics(result, roi)

    def test_invalid_images(self):
        for value in (None, np.zeros((0, 2, 3), np.uint8), self.rgb.astype(float), self.rgb[..., 0]):
            with self.assertRaises(ValueError):
                survey.convert_features(value, 'RGB')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.png'
            path.write_text('not an image')
            with self.assertRaises(ValueError):
                survey.read_image(path)
            with self.assertRaises(ValueError):
                survey.read_image(Path(temp) / 'missing.png')

    def test_empty_or_invalid_roi(self):
        for roi in (np.zeros_like(self.roi), np.zeros((1, 1)), np.full(self.roi.shape, np.nan)):
            with self.assertRaises(ValueError):
                survey.validate_roi(self.rgb, roi)

    def test_outputs_csv_medians_adapter_and_seed(self):
        before = self.rgb.copy(), self.roi.copy()
        self.rgb.setflags(write=False)
        self.roi.setflags(write=False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for init in ('random', 'k-means++'):
                with patch.object(survey, 'segment_leaf', wraps=survey.segment_leaf) as adapter:
                    rows = survey.run_survey(self.rgb, root / init, self.roi,
                                             init=init, repeats=2, max_iter=17, tol=.01)
                self.assertEqual(adapter.call_count, 24)
                for call in adapter.call_args_list:
                    self.assertIs(call.args[0], self.rgb)
                    self.assertEqual(call.kwargs['max_iter'], 17)
                    self.assertEqual(call.kwargs['tol'], .01)
                self.assertEqual(len(rows), 12)
                with (root / init / 'results.csv').open() as stream:
                    raw = list(csv.DictReader(stream))
                self.assertEqual(len(raw), 24)
                required = {'image_id', 'color_space', 'k', 'init', 'random_state',
                            'image_height', 'image_width', 'roi_pixel_count', 'n_iter',
                            'kmeans_exec_time_ms', 'lesion_exec_time_ms', 'adapter_exec_time_ms',
                            'inertia', 'disease_pixel_count', 'healthy_pixel_count',
                            'uncertain_pixel_count', 'disease_ratio_percent',
                            'healthy_ratio_percent', 'uncertain_ratio_percent'}
                self.assertTrue(required <= raw[0].keys())
                for i, row in enumerate(rows):
                    self.assertEqual(row['roi_pixel_count'], int(self.roi.sum()))
                    self.assertAlmostEqual(sum(row[n + '_ratio_percent'] for n in
                                               ('disease', 'healthy', 'uncertain')), 100)
                    for key, value in row.items():
                        if isinstance(value, (int, float)):
                            self.assertTrue(np.isfinite(value))
                    for key in ('inertia', 'n_iter', 'disease_pixel_count'):
                        self.assertEqual(raw[2*i][key], raw[2*i+1][key])
                    for key in ('conversion_exec_time_ms', 'kmeans_exec_time_ms',
                                'lesion_exec_time_ms', 'adapter_exec_time_ms'):
                        self.assertAlmostEqual(row[key], (float(raw[2*i][key])+float(raw[2*i+1][key]))/2)
                with (root / init / 'summary.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 12)
                for path in (root / init).rglob('*.png'):
                    image = survey.read_image(path, mask='mask' in path.name or path.name == 'roi.png')
                    expected = (3*(96+28), 4*128) if path.name.startswith('comparison') else (96, 128)
                    self.assertEqual(image.shape[:2], expected)
                for folder in (root / init).iterdir():
                    if folder.is_dir():
                        masks = [survey.read_image(folder / (n + '_mask.png'), True)
                                 for n in ('disease', 'healthy', 'uncertain')]
                        np.testing.assert_array_equal(sum(m.astype(int) for m in masks), self.roi.astype(int)*255)
                snapshot = (root / init / 'results.csv').read_bytes()
                with self.assertRaises(FileExistsError):
                    survey.run_survey(self.rgb, root / init, self.roi)
                self.assertEqual(snapshot, (root / init / 'results.csv').read_bytes())
            # A second run must reproduce complete segmentation and mask artifacts.
            survey.run_survey(self.rgb, root / 'again', self.roi, max_iter=17, tol=.01)
            for path in (root / 'random').rglob('*.png'):
                self.assertEqual(path.read_bytes(), (root / 'again' / path.relative_to(root / 'random')).read_bytes())
        np.testing.assert_array_equal(self.rgb, before[0])
        np.testing.assert_array_equal(self.roi, before[1])

    def test_full_image_mode_explicit(self):
        with tempfile.TemporaryDirectory() as temp, self.assertWarnsRegex(UserWarning, 'background'):
            rows = survey.run_survey(self.rgb, Path(temp) / 'full')
            self.assertTrue(all(r['roi_mode'] == 'full_image' for r in rows))
            self.assertTrue(all(r['roi_pixel_count'] == 96*128 for r in rows))


if __name__ == '__main__':
    unittest.main()
