"""Original-resolution-supported vegetation-drop classification."""
from __future__ import annotations

import numpy as np

NO_DATA, NO_CHANGE, OBSERVED, INFERRED, UNSUPPORTED = 0, 1, 2, 3, 4


def classify(pre_mean, post_mean, pre_std, post_std, parent_pre, parent_post, valid,
             *, k: float = 2.0, parent_drop_threshold: float, scale: int = 4):
    pre_mean, post_mean = np.asarray(pre_mean), np.asarray(post_mean)
    if pre_mean.shape != post_mean.shape or any(np.shape(x) != pre_mean.shape for x in (pre_std, post_std, valid)):
        raise ValueError('Fine-grid inputs must share shape')
    if pre_mean.shape != (parent_pre.shape[0] * scale, parent_pre.shape[1] * scale) or parent_pre.shape != parent_post.shape:
        raise ValueError('Parent grid must align exactly x4')
    if k < 0 or parent_drop_threshold < 0:
        raise ValueError('Thresholds must be nonnegative')
    support = np.repeat(np.repeat(parent_pre - parent_post > parent_drop_threshold, scale, 0), scale, 1)
    sr_support = pre_mean - post_mean > k * np.sqrt(np.square(pre_std) + np.square(post_std))
    good = np.asarray(valid, dtype=bool) & np.isfinite(pre_mean) & np.isfinite(post_mean) & np.isfinite(pre_std) & np.isfinite(post_std)
    classes = np.full(pre_mean.shape, NO_DATA, dtype=np.uint8)
    classes[good] = NO_CHANGE
    classes[good & support & sr_support] = OBSERVED
    classes[good & support & ~sr_support] = INFERRED
    classes[good & ~support & sr_support] = UNSUPPORTED
    stats = {'unsupported_pixels': int(np.count_nonzero(classes == UNSUPPORTED)),
             'observed_pixels': int(np.count_nonzero(classes == OBSERVED)),
             'inferred_pixels': int(np.count_nonzero(classes == INFERRED)),
             'no_data_pixels': int(np.count_nonzero(classes == NO_DATA))}
    return classes, stats
