import numpy as np

from trustsr.evaluate import spectral_rmse, psnr, f1_score, assess_sr, calibrate_k


def test_spectral_error_uses_parent_averages():
    lr = np.full((4, 2, 2), .3, dtype=np.float32)
    sr = np.repeat(np.repeat(lr, 4, -2), 4, -1)
    sr[:, ::2] += .1
    sr[:, 1::2] -= .1
    assert spectral_rmse(lr, sr) < 1e-6


def test_f1_and_psnr_with_known_values():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    prediction = np.array([1, 0, 1, 0], dtype=bool)
    assert f1_score(truth, prediction) == .5
    assert abs(psnr(np.zeros((4, 8, 8)), np.full((4, 8, 8), .1), data_range=1) - 20) < 1e-6


def test_assessment_distinguishes_training_diagnostic_from_heldout():
    lr = np.full((4, 2, 2), .3, dtype=np.float32)
    hr = np.full((4, 8, 8), .3, dtype=np.float32)
    result = assess_sr(lr, hr, hr, split='train')
    assert result['split'] == 'train'
    assert result['held_out'] is False
    assert result['spectral_rmse'] < 1e-6


def test_k_selection_uses_labelled_parent_grid_and_f1():
    positive = np.full((4, 4), .5, dtype=np.float32)
    post = np.full((4, 4), .3, dtype=np.float32)
    standard = np.full((4, 4), .04, dtype=np.float32)
    sample = {'pre_mean': positive, 'post_mean': post, 'pre_std': standard,
              'post_std': standard, 'parent_pre': np.array([[.5]]),
              'parent_post': np.array([[.3]]), 'valid': np.ones((4, 4), bool),
              'labels': np.array([[1]], bool)}
    result = calibrate_k([sample], [2.0, 6.0], parent_drop_threshold=.1)
    assert result['selected_k'] == 2.0
    assert result['scores'][0]['f1'] == 1.0
