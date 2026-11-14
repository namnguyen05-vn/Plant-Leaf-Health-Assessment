"""Direct H2 unit and local integration tests; no pipeline/fallback imports."""
import unittest
import numpy as np
from src.color_clustering import (
    compute_excess_green_index, extract_disease_and_healthy_masks as extract,
)
from src.kmeans import kmeans_scratch


def scene(colors):
    image = np.repeat(np.array(colors, dtype=np.uint8)[None, :, :], 5, axis=0)
    labels = np.repeat(np.arange(len(colors), dtype=np.int32)[None, :], 5, axis=0)
    return image, labels, np.zeros((len(colors), 3), np.float32)


class LesionTests(unittest.TestCase):
    def test_exg(self):
        colors = np.array([[0, 255, 0], [255, 0, 255], [40, 140, 70]], np.uint8)
        result = compute_excess_green_index(colors)
        np.testing.assert_array_equal(result, [510, -510, 170])
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(compute_excess_green_index(colors[0])), 510)

    def test_all_healthy(self):
        r = extract(*scene([[40, 140, 70]]))
        np.testing.assert_array_equal(r['healthy_mask'], 255)
        self.assertEqual(r['disease_cluster_ids'], [])

    def test_brown(self):
        r = extract(*scene([[40, 140, 70], [120, 65, 25]]))
        self.assertEqual(r['disease_cluster_ids'], [1])
        self.assertEqual(r['cluster_features']['reason'][1], 'brown')

    def test_yellow(self):
        r = extract(*scene([[40, 140, 70], [200, 195, 45]]))
        self.assertEqual(r['disease_cluster_ids'], [1])
        self.assertEqual(r['cluster_features']['reason'][1], 'yellow')

    def test_multiple_diseased(self):
        r = extract(*scene([[40, 140, 70], [120, 65, 25], [200, 195, 45]]))
        self.assertEqual(r['disease_cluster_ids'], [1, 2])

    def test_multiple_healthy(self):
        r = extract(*scene([[40, 140, 70], [60, 180, 80], [30, 80, 40]]))
        self.assertEqual(r['healthy_cluster_ids'], [0, 1, 2])

    def test_shadow_glare_and_neutral_are_uncertain(self):
        r = extract(*scene([[15, 25, 15], [230, 240, 225], [130, 130, 130], [50, 50, 180]]))
        self.assertEqual(r['uncertain_cluster_ids'], [0, 1, 2, 3])
        self.assertEqual(r['healthy_cluster_ids'], [])
        self.assertEqual(r['disease_cluster_ids'], [])
        self.assertEqual(r['cluster_features']['reason'].tolist(),
                         ['shadow_or_too_dark', 'glare', 'low_saturation', 'ambiguous_color'])

    def test_partition_and_types(self):
        image, labels, centers = scene([[40, 140, 70], [120, 65, 25], [230, 240, 225]])
        r = extract(image, labels, centers)
        masks = [r[key] for key in ('healthy_mask', 'disease_mask', 'uncertain_mask')]
        for mask in masks:
            self.assertEqual(mask.shape, labels.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 255})
        np.testing.assert_array_equal(sum(m.astype(np.int32) for m in masks), 255)

    def test_roi_controls_means_and_masks(self):
        image, labels, centers = scene([[40, 140, 70], [120, 65, 25]])
        # Same ID outside ROI must not change the in-leaf RGB representative.
        labels[:] = 0
        mask = np.zeros(labels.shape, np.uint8)
        mask[:, 0] = 255
        r = extract(image, labels, centers, mask)
        np.testing.assert_array_equal(r['cluster_features']['rgb_mean'][0], [40, 140, 70])
        self.assertEqual(r['healthy_cluster_ids'], [0])
        for key in ('healthy_mask', 'disease_mask', 'uncertain_mask'):
            np.testing.assert_array_equal(r[key][mask == 0], 0)

    def test_minus_one(self):
        image, labels, centers = scene([[120, 65, 25], [120, 65, 25]])
        labels[:, 0] = -1
        r = extract(image, labels, centers)
        for key in ('healthy_mask', 'disease_mask', 'uncertain_mask'):
            np.testing.assert_array_equal(r[key][:, 0], 0)
        self.assertEqual(r['disease_cluster_ids'], [1])

    def test_centroid_coordinates_are_not_rgb(self):
        image, labels, centers = scene([[40, 140, 70], [120, 65, 25]])
        for centers in (centers, np.full((2, 6), 250., np.float32),
                        np.array([[80., -40., 50.], [60., 30., -20.]]),
                        np.array([[85., 180., 140.], [15., 200., 120.]])):
            r = extract(image, labels, centers)
            self.assertEqual(r['disease_cluster_ids'], [1])
            self.assertEqual(r['healthy_cluster_ids'], [0])

    def test_empty_roi(self):
        image, labels, centers = scene([[120, 65, 25]])
        r = extract(image, labels, centers, np.zeros(labels.shape, bool))
        for key in ('disease_cluster_ids', 'healthy_cluster_ids', 'uncertain_cluster_ids'):
            self.assertEqual(r[key], [])
        for key in ('healthy_mask', 'disease_mask', 'uncertain_mask'):
            self.assertFalse(r[key].any())
        self.assertEqual(r['cluster_features']['status'].tolist(), ['empty'])

    def test_empty_cluster(self):
        image, labels, centers = scene([[40, 140, 70], [120, 65, 25]])
        labels[:] = 0
        r = extract(image, labels, centers)
        self.assertEqual(r['cluster_features']['pixel_count'].tolist(), [10, 0])
        self.assertEqual(r['cluster_features']['status'][1], 'empty')
        np.testing.assert_array_equal(r['cluster_features']['rgb_mean'][1], 0)

    def test_invalid_inputs(self):
        image, labels, centers = scene([[40, 140, 70], [120, 65, 25]])
        cases = [(image.astype(float), labels, centers, None),
                 (image[:, :, 0], labels, centers, None),
                 (image, labels.astype(float), centers, None),
                 (image, np.full_like(labels, -2), centers, None),
                 (image, np.full_like(labels, 2), centers, None),
                 (image, labels[:, :1], centers, None),
                 (image, labels, np.full((2, 3), np.nan), None),
                 (image, labels, np.zeros((2, 0)), None),
                 (image, labels, centers, np.zeros((1, 1))),
                 (image, labels, centers, np.full(labels.shape, np.inf))]
        for args in cases:
            with self.subTest(shapes=[getattr(a, 'shape', None) for a in args]), self.assertRaises(ValueError):
                extract(*args)
        for color in (np.array([1, 2]), np.array([0., np.nan, 0.]), np.array([-1, 2, 3])):
            with self.assertRaises(ValueError):
                compute_excess_green_index(color)

    def test_nonmutation_and_repeatability(self):
        args = list(scene([[40, 140, 70], [120, 65, 25]]))
        args.append(np.ones(args[1].shape, bool))
        originals = [a.copy() for a in args]
        for a in args:
            a.setflags(write=False)
        a, b = extract(*args), extract(*args)
        for key in ('healthy_mask', 'disease_mask', 'uncertain_mask'):
            np.testing.assert_array_equal(a[key], b[key])
        for current, original in zip(args, originals):
            np.testing.assert_array_equal(current, original)

    def test_natural_yellow_limitation(self):
        # Identical color evidence cannot reveal whether yellow is natural.
        r = extract(*scene([[200, 195, 45]]))
        self.assertEqual(r['disease_cluster_ids'], [0])
        self.assertEqual(r['healthy_cluster_ids'], [])

    def test_disease_majority_not_forced_healthy(self):
        r = extract(*scene([[120, 65, 25], [200, 195, 45]]))
        np.testing.assert_array_equal(r['disease_mask'], 255)
        self.assertEqual(r['healthy_cluster_ids'], [])


class LocalIntegrationTests(unittest.TestCase):
    def test_green_leaf_with_spots(self):
        image = np.full((64, 64, 3), 245, np.uint8)
        roi = np.zeros((64, 64), bool)
        roi[4:60, 4:60] = True
        image[roi] = [40, 140, 70]
        image[15:25, 15:25] = [120, 65, 25]
        image[35:47, 35:47] = [200, 195, 45]
        truth = np.zeros(roi.shape, bool)
        truth[15:25, 15:25] = True
        truth[35:47, 35:47] = True
        for channels in (3, 6):
            features = image if channels == 3 else np.concatenate((image, image), axis=2)
            for init in ('random', 'k-means++'):
                km = kmeans_scratch(features, k=3, roi_mask=roi, init=init)
                result = extract(image, km['labels_2d'], km['centroids'], roi)
                # Separate cluster purity from extractor correctness.
                lesion_ids = np.unique(km['labels_2d'][truth])
                self.assertTrue(all(np.all(truth[km['labels_2d'] == cid]) for cid in lesion_ids))
                np.testing.assert_array_equal(result['disease_mask'] > 0, truth)

    def test_separated_dark_lesion_is_uncertain(self):
        image = np.full((24, 24, 3), [40, 140, 70], np.uint8)
        image[8:16, 8:16] = [20, 12, 6]
        km = kmeans_scratch(image, k=2, init='k-means++')
        self.assertNotEqual(km['labels_2d'][0, 0], km['labels_2d'][10, 10])
        result = extract(image, km['labels_2d'], km['centroids'])
        self.assertEqual(result['disease_mask'][10, 10], 0)
        self.assertEqual(result['uncertain_mask'][10, 10], 255)

    def test_mixed_cluster_hides_lesion(self):
        image = np.full((24, 24, 3), [40, 140, 70], np.uint8)
        image[12:] = [70, 190, 80]
        image[2, 2] = [120, 65, 25]
        km = kmeans_scratch(image, k=2, init='k-means++')
        cid = km['labels_2d'][2, 2]
        self.assertGreater(np.sum(km['labels_2d'] == cid), 1)
        result = extract(image, km['labels_2d'], km['centroids'])
        self.assertEqual(result['disease_mask'][2, 2], 0)


if __name__ == '__main__':
    unittest.main()
