"""Direct tests; run with python -B -m unittest tests.test_segmentation_kmeans -v."""

import unittest
import numpy as np
from src.segmentation.kmeans import (
    initialize_centroids, compute_euclidean_distances, kmeans_scratch,
)


# Frozen H1 source: independent before/after correctness and timing reference.
_H1_SOURCE = r'''"""NumPy Lloyd K-means; no color conversion or pipeline fallback.

Images use uint8 feature channels (RGB or arbitrary C, including C=6).
ROI membership is mask > 0. Empty ROI and N < K raise ValueError.
Duplicate colors are allowed: unused centers retain their previous position.
"""

from time import perf_counter
import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _features(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name} must have nonempty shape (N, C)")
    if value.dtype.kind not in "uif" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real numbers")
    # Float64 intermediates avoid overflow/cancellation for public distance inputs.
    result = value.astype(np.float64)
    if np.max(np.abs(result)) > np.sqrt(np.finfo(np.float32).max / (4 * value.shape[1])):
        raise ValueError(f"{name} values exceed the supported float32 distance range")
    return result


def initialize_centroids(features, k, random_state=42, init="random"):
    """Return (k,C) float32 centers; init is 'random' or 'k-means++'.

    Random selects distinct sample indices, not necessarily distinct colors.
    K-means++ samples with probability proportional to squared distance.
    A local generator leaves NumPy's global random state unchanged.
    """
    x = _features(features, "features")
    _integer(k, "k", 2)
    _integer(random_state, "random_state", 0)
    if k > len(x):
        raise ValueError("number of ROI samples must be >= k")
    if init not in ("random", "k-means++"):
        raise ValueError("init must be 'random' or 'k-means++'")
    rng = np.random.default_rng(random_state)
    if init == "random":
        return x[rng.choice(len(x), k, replace=False)].astype(np.float32)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(len(x))]
    nearest = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = nearest.sum()
        index = rng.choice(len(x), p=nearest / total) if total > 0 else rng.integers(len(x))
        centers[j] = x[index]
        nearest = np.minimum(nearest, np.sum((x - centers[j]) ** 2, axis=1))
    return centers


def compute_euclidean_distances(features, centroids):
    """Return actual Euclidean distances (N,K), float32, via broadcasting."""
    x = _features(features, "features")
    centers = _features(centroids, "centroids")
    if x.shape[1] != centers.shape[1]:
        raise ValueError("features and centroids must have the same channel count")
    diff = x[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def _assign(x, centers):
    # Loop over K, never pixels; avoids an (N,K,C) temporary per iteration.
    distances = np.empty((len(x), len(centers)), dtype=np.float32)
    for j, center in enumerate(centers):
        diff = x - center
        distances[:, j] = np.einsum("ij,ij->i", diff, diff)
    return distances.argmin(axis=1).astype(np.int32)


def kmeans_scratch(img, k=3, max_iter=100, tol=1e-3, random_state=42,
                   roi_mask=None, init="random"):
    """Cluster uint8 (H,W) or (H,W,C) without modifying inputs.

    Retains the legacy positional arguments; adds optional init at the end.
    Stop when maximum Euclidean center displacement <= tol, or max_iter.
    Empty clusters keep their old center (finite and deterministic even for
    constant images). Ties choose the lowest cluster ID. A final assignment
    always makes labels nearest to returned centers, even at the iteration
    limit; at that limit centers need not be means of the final labels.

    Returns segmented_img (uint8, input shape), labels_2d (int32, -1 outside
    ROI), centroids (float32, KxC), n_iter (int), exec_time_ms (float).
    Segmented values are rounded centroid coordinates, not RGB conversion.
    """
    start = perf_counter()
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or 0 in img.shape or img.dtype != np.uint8:
        raise ValueError("img must be a nonempty uint8 (H,W) or (H,W,C) array")
    _integer(k, "k", 2)
    _integer(max_iter, "max_iter", 1)
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and >= 0")
    valid = np.ones(img.shape[:2], dtype=bool)
    if roi_mask is not None:
        if not isinstance(roi_mask, np.ndarray) or roi_mask.shape != img.shape[:2] or roi_mask.dtype.kind not in "buif" or not np.isfinite(roi_mask).all():
            raise ValueError("roi_mask must be a finite numeric/bool (H,W) array")
        valid = roi_mask > 0
    channels = 1 if img.ndim == 2 else img.shape[2]
    x = img[valid].reshape(-1, channels).astype(np.float32)
    if len(x) < k:
        raise ValueError("number of ROI samples must be >= k (empty ROI is invalid)")
    centers = initialize_centroids(x, k, random_state, init)
    for iteration in range(1, max_iter + 1):
        labels = _assign(x, centers)
        counts = np.bincount(labels, minlength=k)
        updated = centers.copy()
        occupied = counts > 0
        for c in range(channels):
            sums = np.bincount(labels, weights=x[:, c], minlength=k)
            updated[occupied, c] = sums[occupied] / counts[occupied]
        shift = np.max(np.linalg.norm(updated - centers, axis=1))
        centers = updated
        if shift <= tol:
            break
    labels = _assign(x, centers)
    labels_2d = np.full(img.shape[:2], -1, dtype=np.int32)
    labels_2d[valid] = labels
    segmented = np.zeros_like(img)
    values = np.clip(np.rint(centers[labels]), 0, 255).astype(np.uint8)
    segmented[valid] = values[:, 0] if img.ndim == 2 else values
    return dict(segmented_img=segmented, labels_2d=labels_2d, centroids=centers,
                n_iter=iteration, exec_time_ms=(perf_counter() - start) * 1000.0)
'''

def h1_reference():
    namespace = {}
    exec(compile(_H1_SOURCE, '<frozen-H1>', 'exec'), namespace)
    return namespace['kmeans_scratch']


class KMeansTests(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((8, 10, 3), dtype=np.uint8)
        self.img[:, 5:] = [210, 120, 60]

    def test_two_colors_and_initializers(self):
        for init in ("random", "k-means++"):
            with self.subTest(init=init):
                # Seed 42 starts in separate colors for this input.
                result = kmeans_scratch(self.img, k=2, init=init)
                np.testing.assert_array_equal(result['segmented_img'], self.img)
                self.assertEqual(len(np.unique(result['labels_2d'])), 2)
                self.assertEqual(result['centroids'].dtype, np.float32)
                self.assertEqual(result['labels_2d'].dtype, np.int32)
                self.assertEqual(result['segmented_img'].dtype, np.uint8)
                self.assertGreater(result['exec_time_ms'], 0)

    def test_seed_and_global_rng(self):
        image = np.random.default_rng(7).integers(0, 256, (12, 13, 3), dtype=np.uint8)
        before = np.random.get_state()
        for init in ('random', 'k-means++'):
            a = kmeans_scratch(image, k=4, init=init)
            b = kmeans_scratch(image, k=4, init=init)
            for key in ('centroids', 'labels_2d', 'segmented_img'):
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a['n_iter'], b['n_iter'])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_random_centers_are_distinct_samples(self):
        features = np.arange(30, dtype=np.float32).reshape(10, 3)
        centers = initialize_centroids(features, 5)
        self.assertEqual(len(np.unique(centers, axis=0)), 5)
        self.assertTrue(all(any(np.array_equal(c, x) for x in features) for c in centers))

    def test_plus_plus_separates_two_colors(self):
        centers = initialize_centroids(self.img.reshape(-1, 3), 2, init='k-means++')
        self.assertEqual(len(np.unique(centers, axis=0)), 2)

    def test_euclidean(self):
        x = np.array([[0, 0], [3, 4]], dtype=np.float32)
        centers = np.array([[0, 4], [3, 0]], dtype=np.float32)
        d = compute_euclidean_distances(x, centers)
        np.testing.assert_allclose(d, [[4, 3], [3, 4]])
        self.assertEqual(d.dtype, np.float32)

    def test_convergence_and_centroid_means(self):
        result = kmeans_scratch(self.img, k=2, max_iter=50, tol=0, init='k-means++')
        self.assertLess(result['n_iter'], 50)
        for j, center in enumerate(result['centroids']):
            np.testing.assert_allclose(center, self.img[result['labels_2d'] == j].mean(axis=0))

    def test_tolerance_stops_early(self):
        result = kmeans_scratch(self.img, k=2, tol=1e6)
        self.assertEqual(result['n_iter'], 1)

    def test_roi_and_nonmutation(self):
        for dtype in (bool, np.uint8):
            mask = np.zeros(self.img.shape[:2], dtype=dtype)
            mask[2:6, 2:8] = True if dtype is bool else 255
            original, mask_original = self.img.copy(), mask.copy()
            result = kmeans_scratch(self.img, k=2, roi_mask=mask, init='k-means++')
            np.testing.assert_array_equal(result['labels_2d'][mask == 0], -1)
            np.testing.assert_array_equal(result['segmented_img'][mask == 0], 0)
            self.assertTrue(np.all(result['labels_2d'][mask > 0] >= 0))
            np.testing.assert_array_equal(original, self.img)
            np.testing.assert_array_equal(mask_original, mask)

    def test_channels_and_grayscale(self):
        for image in (self.img, np.concatenate([self.img, self.img], axis=2), self.img[:, :, 0]):
            result = kmeans_scratch(image, k=2, init='k-means++')
            np.testing.assert_array_equal(result['segmented_img'], image)
            self.assertEqual(result['centroids'].shape, (2, 1 if image.ndim == 2 else image.shape[2]))

    def test_empty_clusters_duplicate_colors(self):
        for init in ('random', 'k-means++'):
            image = np.full((4, 4, 6), 73, dtype=np.uint8)
            result = kmeans_scratch(image, k=5, init=init, tol=0)
            self.assertEqual(result['n_iter'], 1)
            self.assertTrue(np.isfinite(result['centroids']).all())
            np.testing.assert_array_equal(result['segmented_img'], image)
            np.testing.assert_array_equal(result['labels_2d'], 0)

    def test_small_n_and_empty_roi(self):
        with self.assertRaises(ValueError):
            kmeans_scratch(self.img[:1, :1], k=2)
        with self.assertRaises(ValueError):
            initialize_centroids(np.zeros((1, 3)), 2)
        with self.assertRaises(ValueError):
            kmeans_scratch(self.img, roi_mask=np.zeros((8, 10), dtype=bool))
        with self.assertRaises(ValueError):
            kmeans_scratch(self.img, roi_mask=np.eye(8, 10, dtype=bool), k=9)

    def test_invalid_parameters(self):
        for kwargs in ({'k': 1}, {'k': 2.5}, {'k': True}, {'max_iter': 0},
                       {'max_iter': 1.5}, {'tol': -1}, {'tol': float('nan')},
                       {'tol': float('inf')}, {'tol': 'x'}, {'init': 'invalid'},
                       {'random_state': -1}, {'random_state': None},
                       {'roi_mask': np.ones((8, 10, 1))},
                       {'roi_mask': np.full((8, 10), np.nan)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kmeans_scratch(self.img, **kwargs)
        for image in (np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 0), np.uint8),
                      self.img.astype(float), np.zeros(3, np.uint8), None):
            with self.assertRaises(ValueError):
                kmeans_scratch(image)

    def test_invalid_distance_inputs(self):
        for x, centers in ((np.zeros((2, 3)), np.zeros((2, 4))),
                           (np.array([[np.nan]]), np.zeros((2, 1))),
                           (np.zeros((0, 3)), np.zeros((2, 3))),
                           (np.array([[1e38]]), np.zeros((2, 1)))):
            with self.assertRaises(ValueError):
                compute_euclidean_distances(x, centers)

    def test_final_labels_after_iteration_limit(self):
        image = np.random.default_rng(12).integers(0, 256, (9, 11, 6), dtype=np.uint8)
        original = image.copy()
        for init in ('random', 'k-means++'):
            result = kmeans_scratch(image, k=5, max_iter=1, init=init)
            expected = compute_euclidean_distances(image.reshape(-1, 6), result['centroids']).argmin(axis=1)
            np.testing.assert_array_equal(result['labels_2d'].ravel(), expected)
            self.assertEqual(result['n_iter'], 1)
        np.testing.assert_array_equal(image, original)

    def test_helpers_do_not_modify_features(self):
        x = np.arange(24, dtype=np.float32).reshape(8, 3)
        original = x.copy()
        centers = initialize_centroids(x, 3, init='k-means++')
        old_centers = centers.copy()
        compute_euclidean_distances(x, centers)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(centers, old_centers)

    def test_quality_against_frozen_h1(self):
        baseline = h1_reference()
        for channels in (3, 6):
            for seed in (7, 42, 2026):
                image = np.random.default_rng(seed).integers(0, 256, (25, 27, channels), dtype=np.uint8)
                for init in ('random', 'k-means++'):
                    for k in (2, 3, 4, 5):
                        with self.subTest(C=channels, seed=seed, init=init, k=k):
                            a = baseline(image, k=k, init=init, random_state=seed)
                            b = kmeans_scratch(image, k=k, init=init, random_state=seed)
                            self.assertLessEqual(inertia(image, b), inertia(image, a) * 1.001 + 1e-8)
                            self.assertTrue(np.isfinite(b['centroids']).all())
                            expected = compute_euclidean_distances(image.reshape(-1, channels), b['centroids']).argmin(axis=1)
                            np.testing.assert_array_equal(b['labels_2d'].ravel(), expected)

    def test_noncontiguous_readonly_roi(self):
        baseline = h1_reference()
        for channels in (3, 6):
            image = np.random.default_rng(10).integers(0, 256, (21, 23, channels), dtype=np.uint8)[::2, ::2]
            mask = np.indices(image.shape[:2]).sum(axis=0) % 2 == 0
            original = image.copy()
            image.setflags(write=False)
            mask.setflags(write=False)
            for init in ('random', 'k-means++'):
                a = baseline(image, k=5, roi_mask=mask, init=init)
                b = kmeans_scratch(image, k=5, roi_mask=mask, init=init)
                c = kmeans_scratch(image, k=5, roi_mask=mask, init=init)
                self.assertLessEqual(inertia(image, b), inertia(image, a) * 1.001)
                np.testing.assert_array_equal(b['labels_2d'], c['labels_2d'])
                np.testing.assert_array_equal(b['labels_2d'][~mask], -1)
                np.testing.assert_array_equal(b['segmented_img'][~mask], 0)
                self.assertTrue(np.isfinite(b['centroids']).all())
            np.testing.assert_array_equal(image, original)

    def test_exact_tie_selects_first_center(self):
        from src.segmentation.kmeans import _assign
        channels = np.array([[0, 1, 2]], dtype=np.float32)
        centers = np.array([[0], [2]], dtype=np.float32)
        work = (np.empty(3, np.float32), np.empty(3, np.float32),
                np.empty(3, np.float32), np.empty(3, np.int32), np.empty(3, bool))
        # The middle sample is exactly equidistant; repeat with dirty buffers.
        for _ in range(2):
            np.testing.assert_array_equal(_assign(channels, centers, work), [0, 0, 1])


def inertia(image, result, roi_mask=None):
    valid = result['labels_2d'] >= 0
    x = image[valid].reshape(-1, result['centroids'].shape[1]).astype(np.float64)
    diff = x - result['centroids'][result['labels_2d'][valid]]
    return float(np.sum(diff * diff))


def run_benchmark():
    """32 configurations, full warm-up per implementation, 3 paired timed runs.

    python -B -m tests.test_segmentation_kmeans --benchmark
    Times exclude inertia calculation. Emits CSV to stdout; never writes data.
    """
    import os
    import platform
    from time import perf_counter
    baseline = h1_reference()
    print(platform.platform(), 'Python', platform.python_version(),
          'NumPy', np.__version__, 'CPU', platform.processor(),
          'logical CPUs', os.cpu_count(), flush=True)
    print('uniform uint8; data seed=2026; algorithm seed=42; max_iter=100; tol=.001; full ROI; repeats=3')
    print('size,C,init,K,before_ms,after_ms,improvement_pct,before_iter,after_iter,before_inertia,after_inertia,label_disagreement_pct', flush=True)
    for size in (256, 512):
        for channels in (3, 6):
            image = np.random.default_rng(2026).integers(0, 256, (size, size, channels), dtype=np.uint8)
            for init in ('random', 'k-means++'):
                for k in (2, 3, 4, 5):
                    kwargs = dict(k=k, init=init, max_iter=100, tol=.001, random_state=42)
                    baseline(image, **kwargs)
                    kmeans_scratch(image, **kwargs)
                    timings = [[], []]
                    results = [None, None]
                    for repeat in range(3):
                        # Alternate execution order to reduce systematic drift.
                        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            fn = baseline if index == 0 else kmeans_scratch
                            t = perf_counter()
                            result = fn(image, **kwargs)
                            timings[index].append((perf_counter() - t) * 1000)
                            results[index] = result
                    before, after = [float(np.median(t)) for t in timings]
                    a, b = results
                    disagreement = np.mean(a['labels_2d'] != b['labels_2d']) * 100
                    print(f"{size},{channels},{init},{k},{before:.2f},{after:.2f},{100*(before-after)/before:.2f},"
                          f"{a['n_iter']},{b['n_iter']},{inertia(image,a):.6f},{inertia(image,b):.6f},{disagreement:.6f}", flush=True)


def run_memory_and_profile():
    """Separate instrumentation from benchmark timing; print, do not save files.

    tracemalloc measures traced allocations, not process RSS or all native
    allocator/BLAS memory. Input images are allocated before tracing starts.
    """
    import cProfile
    import pstats
    import tracemalloc
    baseline = h1_reference()
    for channels in (3, 6):
        image = np.random.default_rng(2026).integers(0, 256, (512, 512, channels), dtype=np.uint8)
        for name, fn in (('H1', baseline), ('H1.1', kmeans_scratch)):
            tracemalloc.start()
            result = fn(image, k=5, init='random')
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f'PEAK {name} C={channels}: {peak / 2**20:.2f} MiB; iterations={result["n_iter"]}', flush=True)
            del result
    image = np.random.default_rng(2026).integers(0, 256, (512, 512, 6), dtype=np.uint8)
    for name, fn in (('H1', baseline), ('H1.1', kmeans_scratch)):
        print('PROFILE', name, flush=True)
        profiler = cProfile.Profile()
        profiler.runcall(fn, image, k=5, init='random')
        pstats.Stats(profiler).sort_stats('tottime').print_stats(10)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--benchmark']:
        run_benchmark()
    elif sys.argv[1:] == ['--profile']:
        run_memory_and_profile()
    else:
        unittest.main()
