"""Independent H1/H2 bridge; no preprocessing, fusion, morphology or fallback.

For H3.2, call with blurred_rgb, leaf_mask and pathology_features already
computed by the pipeline. kmeans_result is its RGB display contract;
kmeans_raw retains every feature channel and original feature centroids.
color_result retains H2's three-way classification before any fusion.
"""

from time import perf_counter
import numpy as np
from .kmeans import kmeans_scratch
from .color_clustering import extract_disease_and_healthy_masks


def segment_leaf(rgb_img, roi_mask=None, features=None, k=4, max_iter=100,
                 tol=1e-3, random_state=42, init='random'):
    """Run new K-means then H2; inputs are read-only from this function's view.

    rgb_img: nonempty uint8 RGB (H,W,3). features: optional uint8 (H,W)
    or (H,W,C), same spatial shape; defaults to RGB. Feature values are never
    assumed to be RGB. ROI: finite numeric/bool (H,W), membership >0; None
    means the caller considers the entire image leaf. Empty ROI/N<K raises
    ValueError consistently with H1. Invalid parameters propagate from H1.

    Top-level five K-means fields preserve raw feature-space meaning.
    Top-level H2 fields preserve disease/healthy/uncertain masks and IDs.
    kmeans_raw exposes the same raw result as a dictionary; kmeans_result
    provides pipeline RGB display keys, with RGB means of final labels as
    centroids (empty clusters use zero). color_result is the complete H2
    dictionary. Arrays are shared between views, not defensive copies.

    exec_time_ms is K-means-only, as in H1. Separate lesion_exec_time_ms and
    adapter_exec_time_ms measure classification and the complete adapter.
    No masks are fused or morphed, and non-disease is never assumed healthy.
    """
    start = perf_counter()
    if (not isinstance(rgb_img, np.ndarray) or rgb_img.ndim != 3
            or rgb_img.shape[2] != 3 or 0 in rgb_img.shape or rgb_img.dtype != np.uint8):
        raise ValueError('rgb_img must be nonempty uint8 RGB (H,W,3)')
    data = rgb_img if features is None else features
    if (not isinstance(data, np.ndarray) or data.ndim not in (2, 3)
            or data.shape[:2] != rgb_img.shape[:2] or 0 in data.shape
            or data.dtype != np.uint8):
        raise ValueError('features must be uint8 (H,W) or (H,W,C) matching RGB')
    raw = kmeans_scratch(data, k=k, max_iter=max_iter, tol=tol,
                         random_state=random_state, roi_mask=roi_mask, init=init)
    color_start = perf_counter()
    color = extract_disease_and_healthy_masks(rgb_img, raw['labels_2d'],
                                              raw['centroids'], leaf_mask=roi_mask)
    lesion_ms = (perf_counter() - color_start) * 1000
    rgb_centers = color['cluster_features']['rgb_mean']
    valid = raw['labels_2d'] >= 0
    rgb_segmented = np.zeros_like(rgb_img)
    rgb_segmented[valid] = np.clip(np.rint(rgb_centers[raw['labels_2d'][valid]]), 0, 255).astype(np.uint8)
    display = dict(raw, segmented_img=rgb_segmented, centroids=rgb_centers)
    return dict(raw, **color, kmeans_raw=raw, kmeans_result=display,
                color_result=color, lesion_exec_time_ms=lesion_ms,
                adapter_exec_time_ms=(perf_counter() - start) * 1000)
