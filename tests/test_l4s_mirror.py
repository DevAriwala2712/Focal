import h5py
import numpy as np
import pytest


def test_mirror_pair_checks_labels_and_only_four_permitted_bands(tmp_path):
    from risk.r4_labels import inspect_l4s_pair
    image, mask = tmp_path / 'image.h5', tmp_path / 'mask.h5'
    data = np.zeros((128, 128, 14))
    data[:, :, 0] = np.nan  # Excluded band must not enter the spectral input.
    labels = np.zeros((128, 128), dtype='uint8')
    labels[:2, :3] = 1
    with h5py.File(image, 'w') as f:
        f['img'] = data
    with h5py.File(mask, 'w') as f:
        f['mask'] = labels
    result = inspect_l4s_pair(image, mask, [3, 2, 1, 7])
    assert result['labelled_pair_verified'] is True
    assert result['landslide_pixels'] == 6
    assert result['selected_shape'] == [128, 128, 4]
    with h5py.File(mask, 'r+') as f:
        f['mask'][0, 0] = 2
    with pytest.raises(ValueError, match='binary'):
        inspect_l4s_pair(image, mask, [3, 2, 1, 7])
    with pytest.raises(ValueError, match='band'):
        inspect_l4s_pair(image, mask, [0, 2, 1, 7])
