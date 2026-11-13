"""NumPy color rules for suspected lesions using RGB means inside the ROI.

Color symptoms do not establish disease causes: natural yellow/brown leaves
can look identical to lesions. Mixed clusters may hide small lesions.
"""

import numpy as np

# RGB, brightness, ExG use 0..255 units; saturation uses 0..1.
MIN_VISIBLE_VALUE = 45.0  # Darker max(R,G,B) has unreliable color.
MIN_COLOR_SATURATION = 0.20  # Reject nearly neutral colors.
GLARE_MIN_CHANNEL = 175.0
GLARE_MIN_VALUE = 220.0
GLARE_MAX_SATURATION = 0.22
HEALTHY_MIN_EXG = 25.0
HEALTHY_MIN_GREEN_RED_RATIO = 1.15
HEALTHY_MIN_GREEN_BLUE_RATIO = 1.10
YELLOW_MIN_RED_GREEN_RATIO = 0.90
YELLOW_MIN_GREEN_RED_RATIO = 0.80
YELLOW_MAX_BLUE_RATIO = 0.65  # B/max(R,G).
YELLOW_MIN_BLUE_DEFICIT = 25.0  # min(R,G)-B: both R and G must exceed B.
YELLOW_MIN_VALUE = 70.0
BROWN_MIN_RED_GREEN_RATIO = 1.05
BROWN_MIN_GREEN_BLUE_RATIO = 1.15
BROWN_MAX_BRIGHTNESS = 180.0


def compute_excess_green_index(rgb):
    """Return float32 2G-R-B for finite RGB (...,3) in 0..255, without overflow."""
    if (not isinstance(rgb, np.ndarray) or rgb.ndim < 1 or rgb.shape[-1] != 3
            or rgb.dtype.kind not in 'uif' or not np.isfinite(rgb).all()
            or np.any(rgb < 0) or np.any(rgb > 255)):
        raise ValueError('rgb must be finite numeric (...,3) in 0..255')
    colors = rgb.astype(np.float32)
    return 2.0 * colors[..., 1] - colors[..., 0] - colors[..., 2]


def extract_disease_and_healthy_masks(rgb_img, labels_2d, centroids, leaf_mask=None):
    """Classify clusters using original RGB, never feature center coordinates.

    Inputs: uint8 RGB (H,W,3), integer labels (H,W) in [-1,K-1], finite
    real centroids (K,C), C>=1, and optional finite numeric/bool ROI (H,W),
    membership >0. HSV/Lab/six-channel centers are valid: only K is used.
    Inputs are not modified. Without ROI, caller asserts all labeled pixels
    belong to the leaf. Invalid labels are rejected even outside ROI.

    Required outputs: disease_mask, healthy_mask (uint8 {0,255}, HxW),
    disease_cluster_ids (list[int]). Additional uncertain_mask and ID lists
    distinguish ambiguity from healthy tissue. Label -1 is excluded from all
    masks. Empty ROI returns zero masks and empty ID lists. cluster_features
    arrays have K rows indexed by ID; empty clusters have zero numerical
    features, status/reason 'empty', and belong to no ID list.
    """
    if (not isinstance(rgb_img, np.ndarray) or rgb_img.ndim != 3
            or rgb_img.shape[2] != 3 or 0 in rgb_img.shape or rgb_img.dtype != np.uint8):
        raise ValueError('rgb_img must be nonempty uint8 RGB (H,W,3)')
    if (not isinstance(centroids, np.ndarray) or centroids.ndim != 2
            or 0 in centroids.shape or centroids.dtype.kind not in 'uif'
            or not np.isfinite(centroids).all()):
        raise ValueError('centroids must be finite real (K,C), K>=1, C>=1')
    k = len(centroids)
    if (not isinstance(labels_2d, np.ndarray) or labels_2d.shape != rgb_img.shape[:2]
            or labels_2d.dtype.kind not in 'iu'
            or np.any(labels_2d < -1) or np.any(labels_2d >= k)):
        raise ValueError('labels_2d must be integer (H,W) in [-1,K-1]')
    valid = labels_2d >= 0
    if leaf_mask is not None:
        if (not isinstance(leaf_mask, np.ndarray) or leaf_mask.shape != valid.shape
                or leaf_mask.dtype.kind not in 'buif' or not np.isfinite(leaf_mask).all()):
            raise ValueError('leaf_mask must be finite numeric/bool (H,W)')
        valid &= leaf_mask > 0
    labels = labels_2d[valid].astype(np.intp)
    pixels = rgb_img[valid]
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    mean_rgb = np.zeros((k, 3), dtype=np.float32)
    occupied = counts > 0
    for c in range(3):
        sums = np.bincount(labels, weights=pixels[:, c], minlength=k)
        np.divide(sums, counts, out=mean_rgb[:, c], where=occupied)
    r, g, b = mean_rgb.T
    value = mean_rgb.max(axis=1)
    minimum = mean_rgb.min(axis=1)
    saturation = np.zeros(k, dtype=np.float32)
    np.divide(value - minimum, value, out=saturation, where=value > 0)
    brightness = mean_rgb.mean(axis=1)
    exg = compute_excess_green_index(mean_rgb)
    red_green_difference = r - g
    blue_deficit = np.minimum(r, g) - b
    shadow = value < MIN_VISIBLE_VALUE
    glare = ((minimum >= GLARE_MIN_CHANNEL) & (value >= GLARE_MIN_VALUE)
             & (saturation <= GLARE_MAX_SATURATION))
    reliable = occupied & ~shadow & ~glare & (saturation >= MIN_COLOR_SATURATION)
    healthy = (reliable & (exg >= HEALTHY_MIN_EXG)
               & (g >= HEALTHY_MIN_GREEN_RED_RATIO * r)
               & (g >= HEALTHY_MIN_GREEN_BLUE_RATIO * b))
    yellow = (reliable & (r >= YELLOW_MIN_RED_GREEN_RATIO * g)
              & (g >= YELLOW_MIN_GREEN_RED_RATIO * r)
              & (b <= YELLOW_MAX_BLUE_RATIO * np.maximum(r, g))
              & (blue_deficit >= YELLOW_MIN_BLUE_DEFICIT)
              & (value >= YELLOW_MIN_VALUE))
    brown = (reliable & (r >= BROWN_MIN_RED_GREEN_RATIO * g)
             & (g >= BROWN_MIN_GREEN_BLUE_RATIO * b)
             & (brightness <= BROWN_MAX_BRIGHTNESS))
    disease = (yellow | brown) & ~healthy
    uncertain = occupied & ~healthy & ~disease
    status = np.full(k, 'empty', dtype='<U10')
    reason = np.full(k, 'empty', dtype='<U24')
    status[uncertain], reason[uncertain] = 'uncertain', 'ambiguous_color'
    reason[uncertain & shadow] = 'shadow_or_too_dark'
    reason[uncertain & glare] = 'glare'
    reason[uncertain & ~shadow & ~glare & (saturation < MIN_COLOR_SATURATION)] = 'low_saturation'
    status[healthy], reason[healthy] = 'healthy', 'green'
    status[disease] = 'disease'
    reason[disease & brown] = 'brown'
    reason[disease & yellow] = 'yellow'

    def mask_for(selection):
        mask = np.zeros(valid.shape, dtype=np.uint8)
        mask[valid] = selection[labels].astype(np.uint8) * 255
        return mask

    return {
        'disease_mask': mask_for(disease), 'healthy_mask': mask_for(healthy),
        'disease_cluster_ids': np.flatnonzero(disease).tolist(),
        'healthy_cluster_ids': np.flatnonzero(healthy).tolist(),
        'uncertain_cluster_ids': np.flatnonzero(uncertain).tolist(),
        'uncertain_mask': mask_for(uncertain),
        'cluster_features': dict(rgb_mean=mean_rgb, pixel_count=counts, exg=exg,
                                 red_green_difference=red_green_difference,
                                 blue_deficit=blue_deficit, brightness=brightness,
                                 saturation=saturation, value=value,
                                 status=status, reason=reason),
    }
