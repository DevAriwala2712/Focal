import hashlib

import h5py
import numpy as np
import pytest

from scripts.evaluate_sen12 import UnusableTemporalPatchError, read_patch


def make_patch(path, event_date='2024-01-03'):
    with h5py.File(path, 'w') as ds:
        ds.attrs['event_date'] = event_date
        time = ds.create_dataset('time', data=np.array([0, 1, 3]))
        time.attrs['units'] = 'days since 2024-01-01'
        for band in ('B02', 'B03', 'B04', 'B08'):
            ds.create_dataset(band, data=np.full((3, 4, 4), 1000, dtype='int16'))
        scl = np.full((3, 4, 4), 4, dtype='int16')
        scl[1] = 9
        ds.create_dataset('SCL', data=scl)
        ds.create_dataset('MASK', data=np.ones((3, 4, 4), dtype='uint8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sen12_selects_clear_dates_around_single_event(tmp_path):
    path = tmp_path/'patch.nc'
    digest = make_patch(path)
    _, _, valid, labels, metadata = read_patch(path, digest, [0, 1, 2, 3, 8, 9, 10, 11], 10)
    assert metadata['pre_date'] == '2024-01-01'
    assert metadata['post_date'] == '2024-01-04'
    assert valid.all() and labels.all()


def test_sen12_rejects_shared_multi_event_mask(tmp_path):
    path = tmp_path/'patch.nc'
    digest = make_patch(path, '2024-01-03,2024-01-04')
    with pytest.raises(UnusableTemporalPatchError, match='Multiple events'):
        read_patch(path, digest, [0, 1, 2, 3, 8, 9, 10, 11], 10)
