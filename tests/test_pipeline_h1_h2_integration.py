"""Integration boundary: retain raw H2 while main owns fused output."""

from unittest.mock import patch

import numpy as np
import pytest

from src import pipeline
from src.color_clustering import extract_disease_and_healthy_masks
from src.kmeans import kmeans_scratch


def test_pipeline_preserves_complete_h2_without_reclassifying_uncertain():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    captured = []

    def classify(*args, **kwargs):
        result = extract_disease_and_healthy_masks(*args, **kwargs)
        captured.append(result)
        return result

    with patch.object(pipeline, 'kmeans_scratch', wraps=kmeans_scratch) as h1, \
            patch.object(pipeline, 'extract_disease_and_healthy_masks', side_effect=classify) as h2:
        result = pipeline.run_full_diagnosis_pipeline(
            image, k_clusters=2, run_benchmark_comparison=False,
        )

    assert h1.call_count == h2.call_count == 1
    assert h1.call_args.args[0].shape == (32, 32, 6)
    assert not {'kmeans', 'color_clustering'} & set(result['fallback_modules'])
    raw = result['cluster_color_result']
    assert raw is captured[0]  # Retain all masks, IDs, reasons and features.
    valid = (result['leaf_mask'] > 0) & (result['kmeans_result']['labels_2d'] >= 0)
    assert np.count_nonzero(raw['uncertain_mask']) == np.count_nonzero(valid)
    for key in ('disease_mask', 'healthy_mask', 'uncertain_mask'):
        mask = raw[key]
        assert mask.dtype == np.uint8 and mask.shape == valid.shape
        assert not np.any(mask[~valid])
    assert not np.any(raw['healthy_mask'])
    assert not np.any(raw['disease_mask'])
    assert np.any(result['color_result']['healthy_mask'])
    assert 'uncertain_mask' not in result['color_result']
    assert 'reason' in raw['cluster_features']
    assert raw['uncertain_cluster_ids']
    np.testing.assert_array_equal(image, np.full_like(image, 128))


def test_pipeline_propagates_strict_h1_error_without_fallback():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with patch.object(pipeline, '_fallback_kmeans') as fallback:
        with pytest.raises(ValueError, match='number of ROI samples must be >= k'):
            pipeline.run_full_diagnosis_pipeline(image, run_benchmark_comparison=False)
    fallback.assert_not_called()
