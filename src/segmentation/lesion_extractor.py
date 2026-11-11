"""H2 interface proposal: classify clusters as healthy or diseased tissue.

No classification algorithm is implemented in H1. Coordinates used for color
rules must be RGB; HSV/Lab feature centroids must not be interpreted as RGB.
"""


def extract_disease_and_healthy_masks(rgb_img, labels_2d, centroids, leaf_mask=None):
    """Proposed adapter-compatible interface, pending H2 review.

    Inputs: RGB uint8 (H,W,3), int32 labels (H,W), float32 RGB centroids
    (K,3), optional uint8/bool leaf mask (H,W). Labels -1 indicate outside ROI.
    Planned output: dict with disease_mask and healthy_mask (uint8 {0,255},
    HxW) and disease_cluster_ids (list[int]). Inputs will remain unchanged.
    Thresholds and handling of HSV/Lab clustering will be decided in H2.
    """
    raise NotImplementedError("Lesion extraction is scheduled for H2")
