"""NumPy Lloyd K-means; no color conversion or pipeline fallback.

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


def _assign(channels, centers, work):
    """Streaming squared distances, O(N) scratch, no (N,K) matrix.

    channels is contiguous (C,N); work is reused across Lloyd iterations.
    Channelwise accumulation can round differently from H1's einsum;
    strict comparison preserves the lowest cluster ID for exact ties.
    """
    best, distance, temporary, labels, better = work
    best.fill(np.inf)
    labels.fill(0)
    for j, center in enumerate(centers):
        np.subtract(channels[0], center[0], out=distance)
        np.square(distance, out=distance)
        for c in range(1, len(channels)):
            np.subtract(channels[c], center[c], out=temporary)
            np.square(temporary, out=temporary)
            np.add(distance, temporary, out=distance)
        np.less(distance, best, out=better)
        np.copyto(best, distance, where=better)
        np.copyto(labels, j, where=better)
    return labels


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
    valid = None
    if roi_mask is not None:
        if not isinstance(roi_mask, np.ndarray) or roi_mask.shape != img.shape[:2] or roi_mask.dtype.kind not in "buif" or not np.isfinite(roi_mask).all():
            raise ValueError("roi_mask must be a finite numeric/bool (H,W) array")
        valid = roi_mask > 0
    channels = 1 if img.ndim == 2 else img.shape[2]
    pixels = img.reshape(-1, channels) if valid is None else img[valid].reshape(-1, channels)
    x = pixels.astype(np.float32)
    if len(x) < k:
        raise ValueError("number of ROI samples must be >= k (empty ROI is invalid)")
    centers = initialize_centroids(x, k, random_state, init)
    channel_data = np.ascontiguousarray(x.T)
    # bincount otherwise converts strided float32 weights to float64 every
    # iteration; prepare contiguous float64 weights once instead.
    weights = channel_data.astype(np.float64)
    del x
    n = channel_data.shape[1]
    work = (np.empty(n, np.float32), np.empty(n, np.float32),
            np.empty(n, np.float32), np.empty(n, np.int32), np.empty(n, bool))
    for iteration in range(1, max_iter + 1):
        labels = _assign(channel_data, centers, work)
        counts = np.bincount(labels, minlength=k)
        updated = centers.copy()
        occupied = counts > 0
        for c in range(channels):
            sums = np.bincount(labels, weights=weights[c], minlength=k)
            updated[occupied, c] = sums[occupied] / counts[occupied]
        shift = np.max(np.linalg.norm(updated - centers, axis=1))
        centers = updated
        if shift <= tol:
            break
    labels = _assign(channel_data, centers, work)
    labels_2d = np.full(img.shape[:2], -1, dtype=np.int32)
    if valid is None:
        labels_2d[:] = labels.reshape(img.shape[:2])
    else:
        labels_2d[valid] = labels
    segmented = np.zeros_like(img)
    values = np.clip(np.rint(centers[labels]), 0, 255).astype(np.uint8)
    if valid is None:
        segmented[:] = values.reshape(img.shape)
    else:
        segmented[valid] = values[:, 0] if img.ndim == 2 else values
    return dict(segmented_img=segmented, labels_2d=labels_2d, centroids=centers,
                n_iter=iteration, exec_time_ms=(perf_counter() - start) * 1000.0)
