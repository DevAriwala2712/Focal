import numpy as np

from trustsr.change import classify, NO_DATA, NO_CHANGE, OBSERVED, INFERRED, UNSUPPORTED


def test_change_classes_parent_support_artifact_and_mask():
    pre = np.full((8, 8), .7, dtype=np.float32)
    post = pre.copy()
    post[:4, :4] = .2
    post[4:, :4] = .4
    post[:4, 4:] = .2
    std = np.zeros_like(pre)
    std[4:, :4] = .3
    parent_pre = np.full((2, 2), .7, dtype=np.float32)
    parent_post = parent_pre.copy()
    parent_post[0, 0] = .2
    parent_post[1, 0] = .4
    valid = np.ones((8, 8), dtype=bool)
    valid[7, 7] = False
    got, stats = classify(pre, post, std, std, parent_pre, parent_post, valid,
                          k=2, parent_drop_threshold=.1, scale=4)
    assert got[0, 0] == OBSERVED
    assert got[5, 0] == INFERRED
    assert got[0, 5] == UNSUPPORTED
    assert got[5, 5] == NO_CHANGE
    assert got[7, 7] == NO_DATA
    assert stats['unsupported_pixels'] == 16
