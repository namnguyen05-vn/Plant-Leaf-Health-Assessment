"""Direct adapter tests; --compare runs the existing pipeline separately."""
import unittest
from unittest.mock import patch
import numpy as np
from src.adapter import segment_leaf
from src.kmeans import kmeans_scratch
from src.color_clustering import extract_disease_and_healthy_masks


def make_scene(kind='mixed'):
    image = np.full((96, 96, 3), 240, np.uint8)
    roi = np.zeros((96, 96), bool)
    roi[8:88, 8:88] = True
    image[roi] = [40, 140, 70]
    truth = np.zeros(roi.shape, bool)
    if kind in ('yellow', 'mixed'):
        image[25:38, 25:38] = [200, 195, 45]
        truth[25:38, 25:38] = True
    if kind in ('brown', 'mixed'):
        image[55:68, 55:68] = [120, 65, 25]
        truth[55:68, 55:68] = True
    if kind == 'dark':
        image[25:38, 25:38] = [20, 12, 6]
    if kind == 'glare':
        image[25:38, 25:38] = [230, 240, 225]
    return image, roi, truth


class AdapterTests(unittest.TestCase):
    def test_scenes_channels_initializers(self):
        for kind in ('healthy', 'yellow', 'brown', 'mixed'):
            image, roi, truth = make_scene(kind)
            for c in (3, 6):
                features = image if c == 3 else np.concatenate([image, image], axis=2)
                for init in ('random', 'k-means++'):
                    with self.subTest(kind=kind, C=c, init=init):
                        result = segment_leaf(image, roi, features, k=4, init=init)
                        np.testing.assert_array_equal(result['disease_mask'] > 0, truth)
                        np.testing.assert_array_equal(result['healthy_mask'] > 0, roi & ~truth)
                        self.assertEqual(result['segmented_img'].shape, features.shape)
                        self.assertEqual(result['centroids'].shape, (4, c))
                        self.assertEqual(result['kmeans_result']['segmented_img'].shape, image.shape)
                        self.assertEqual(result['kmeans_result']['centroids'].shape, (4, 3))

    def test_uncertain_is_not_healthy(self):
        for kind in ('dark', 'glare'):
            image, roi, _ = make_scene(kind)
            r = segment_leaf(image, roi, k=2, init='k-means++')
            self.assertEqual(r['uncertain_mask'][30, 30], 255)
            self.assertEqual(r['healthy_mask'][30, 30], 0)
            self.assertEqual(r['disease_mask'][30, 30], 0)

    def test_mask_contract(self):
        image, roi, _ = make_scene('glare')
        r = segment_leaf(image, roi, k=3, init='k-means++')
        total = np.zeros(roi.shape, np.int32)
        for key in ('disease_mask', 'healthy_mask', 'uncertain_mask'):
            mask = r[key]
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 255})
            np.testing.assert_array_equal(mask[~roi], 0)
            total += mask
        np.testing.assert_array_equal(total, roi.astype(np.int32) * 255)
        np.testing.assert_array_equal(r['labels_2d'][~roi], -1)
        self.assertEqual(r['labels_2d'].dtype, np.int32)
        self.assertEqual(r['centroids'].dtype, np.float32)

    def test_seed_nonmutation_and_no_features(self):
        image, roi, _ = make_scene()
        features = np.concatenate([image, image], axis=2)
        originals = [x.copy() for x in (image, roi, features)]
        for x in (image, roi, features):
            x.setflags(write=False)
        for init in ('random', 'k-means++'):
            a = segment_leaf(image, roi, features, init=init)
            b = segment_leaf(image, roi, features, init=init)
            for key in ('labels_2d', 'centroids', 'disease_mask', 'healthy_mask', 'uncertain_mask'):
                np.testing.assert_array_equal(a[key], b[key])
        for x, before in zip((image, roi, features), originals):
            np.testing.assert_array_equal(x, before)
        a = segment_leaf(image, roi)
        b = segment_leaf(image, roi, features=image)
        np.testing.assert_array_equal(a['labels_2d'], b['labels_2d'])

    def test_exact_delegation_and_timings(self):
        image, roi, _ = make_scene()
        with patch('src.adapter.kmeans_scratch', wraps=kmeans_scratch) as km, patch(
                'src.adapter.extract_disease_and_healthy_masks',
                wraps=extract_disease_and_healthy_masks) as ex:
            r = segment_leaf(image, roi, k=3, max_iter=7, tol=.02, random_state=8, init='k-means++')
            self.assertEqual(km.call_count, 1)
            self.assertEqual(ex.call_count, 1)
            self.assertEqual(km.call_args.kwargs['max_iter'], 7)
            self.assertEqual(km.call_args.kwargs['random_state'], 8)
            self.assertEqual(km.call_args.kwargs['tol'], .02)
        self.assertGreater(r['exec_time_ms'], 0)
        self.assertGreater(r['lesion_exec_time_ms'], 0)
        self.assertGreaterEqual(r['adapter_exec_time_ms'], r['exec_time_ms'] + r['lesion_exec_time_ms'])
        for key in ('segmented_img', 'labels_2d', 'centroids', 'n_iter', 'exec_time_ms'):
            self.assertIn(key, r['kmeans_result'])
        for key in ('disease_cluster_ids', 'healthy_cluster_ids', 'uncertain_cluster_ids'):
            self.assertIsInstance(r[key], list)

    def test_feature_channels_not_assumed_rgb(self):
        image, roi, _ = make_scene()
        features = image[:, :, ::-1].copy()
        r = segment_leaf(image, roi, features, k=3, init='k-means++')
        np.testing.assert_array_equal(r['disease_mask'] > 0, make_scene()[2])
        for cid in np.unique(r['labels_2d'][roi]):
            expected = image[r['labels_2d'] == cid].mean(axis=0)
            np.testing.assert_allclose(r['kmeans_result']['centroids'][cid], expected, atol=1e-5)
        self.assertFalse(np.array_equal(r['centroids'], r['kmeans_result']['centroids']))

    def test_errors(self):
        image, roi, _ = make_scene()
        for kwargs in ({'features': np.zeros((1, 1, 6), np.uint8)},
                       {'features': image.astype(float)}, {'roi_mask': np.zeros((1, 1))},
                       {'roi_mask': np.zeros_like(roi)}, {'roi_mask': np.full(roi.shape, np.nan)},
                       {'k': 10000}, {'k': 1}, {'init': 'bad'}, {'max_iter': 0}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                segment_leaf(image, **kwargs)
        with self.assertRaises(ValueError):
            segment_leaf(image.astype(float))


def compare_pipeline():
    """Diagnostic only: pipeline may use its existing fallbacks; adapter never does.

    Share pipeline preprocessing/ROI/features for a controlled stage comparison.
    Adapter timings exclude preprocessing, fusion and postprocessing. Print only.
    """
    import json
    from itertools import permutations
    from time import perf_counter
    from src.pipeline import (run_full_diagnosis_pipeline, build_pathology_feature_space,
                              call_color_clustering, detect_pathology_disease_mask)
    from src.postprocess import closing_scratch, opening_scratch, connected_components_bfs_scratch
    def stats(mask, roi):
        count = int(np.sum((mask > 0) & (roi > 0)))
        return [count, round(100 * count / max(1, int(np.sum(roi > 0))), 3)]
    def fit_sse(features, labels, roi):
        total = 0.
        for cid in np.unique(labels[roi > 0]):
            x = features[(labels == cid) & (roi > 0)].astype(np.float64)
            total += float(np.sum((x - x.mean(axis=0)) ** 2))
        return round(total, 2)
    for kind in ('healthy', 'yellow', 'brown', 'mixed', 'dark', 'glare'):
        image, _, _ = make_scene(kind)
        t = perf_counter()
        a = run_full_diagnosis_pipeline(image, k_clusters=4, blur_kernel=3,
                                       morph_kernel=3, min_lesion_area=5, run_benchmark_comparison=False)
        pipeline_ms = (perf_counter() - t) * 1000
        rgb, roi = a['blurred'], a['leaf_mask']
        features = build_pathology_feature_space(rgb, a['hsv'])
        b = segment_leaf(rgb, roi, features, k=4, init='k-means++')
        ac = call_color_clustering(rgb, a['kmeans_result']['labels_2d'],
                                   a['kmeans_result']['centroids'], leaf_mask=roi)
        # Separate downstream diagnostic, not part of adapter behavior/output.
        bf = detect_pathology_disease_mask(rgb, roi, b['disease_mask'])
        bm = connected_components_bfs_scratch(opening_scratch(
            closing_scratch(bf['disease_mask'], kernel_size=3), kernel_size=3), min_area=5)['cleaned_mask']
        la = a['kmeans_result']['labels_2d'][roi > 0]
        lb = b['labels_2d'][roi > 0]
        best = max(np.mean(la == np.asarray(p)[lb]) for p in permutations(range(4)))
        union = np.sum((ac['disease_mask'] > 0) | (b['disease_mask'] > 0))
        inter = np.sum((ac['disease_mask'] > 0) & (b['disease_mask'] > 0))
        print(json.dumps(dict(scene=kind, roi_px=int(np.sum(roi > 0)),
            fallback=a['fallback_modules'], cluster_agreement_pct=round(100*best, 3),
            fit_sse_A=fit_sse(features, a['kmeans_result']['labels_2d'], roi), fit_sse_B=fit_sse(features, b['labels_2d'], roi),
            iter_A=a['kmeans_result']['n_iter'], iter_B=b['n_iter'],
            disease_A_cluster=stats(ac['disease_mask'], roi), disease_A_fused=stats(a['color_result']['disease_mask'], roi),
            disease_A_post=stats(a['postprocess_result']['cleaned_mask'], roi), disease_B_cluster=stats(b['disease_mask'], roi),
            disease_B_fused_diagnostic=stats(bf['disease_mask'], roi), disease_B_post_diagnostic=stats(bm, roi),
            healthy_A_cluster=stats(ac['healthy_mask'], roi), healthy_A_fused=stats(a['color_result']['healthy_mask'], roi),
            healthy_B_cluster=stats(b['healthy_mask'], roi), uncertain_B=stats(b['uncertain_mask'], roi),
            disease_cluster_iou=round(float(inter / union), 4) if union else 1.,
            pipeline_total_ms=round(pipeline_ms, 2), km_A_ms=round(a['kmeans_result']['exec_time_ms'], 2),
            adapter_ms=round(b['adapter_exec_time_ms'], 2), km_B_ms=round(b['exec_time_ms'], 2)), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--compare']:
        compare_pipeline()
    else:
        unittest.main()
