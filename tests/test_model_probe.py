import pytest


def test_model_artifact_mismatch_is_rejected_before_import(tmp_path):
    from risk.model import verify_artifacts
    (tmp_path / 'load.py').write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match='hash'):
        verify_artifacts(tmp_path, {'load.py': '0' * 64})


def test_training_path_reaches_pretrained_convolution_weights():
    import torch
    from sen2sr.models.opensr_baseline.cnn import CNNSR
    from risk.model import enable_training_path
    torch.set_num_threads(1)
    torch.manual_seed(2)
    sr = CNNSR(4, 4, 4, 4, True, False, 1)
    x = torch.rand(1, 4, 8, 8)
    before = sr(x).detach()
    enable_training_path(sr)
    after = sr(x)
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)
    after.mean().backward()
    assert sr.conv_1.conv[0].weight.grad is not None
    assert sr.conv_1.conv[0].weight.grad.abs().sum() > 0


def test_label_inventory_inspection_reads_features(tmp_path):
    import sqlite3
    from risk.r4_labels import inspect_inventory
    path = tmp_path / 'labels.gpkg'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT, srs_id INTEGER, min_x REAL, min_y REAL, max_x REAL, max_y REAL)')
        db.execute("INSERT INTO gpkg_contents VALUES ('slides','features',4326,0,0,1,1)")
        db.execute('CREATE TABLE slides (fid INTEGER, event_date TEXT)')
        db.execute("INSERT INTO slides VALUES (1,'2024-07-30')")
    result = inspect_inventory(path)
    assert result['feature_count'] == 1
    assert result['event_dates'] == ['2024-07-30']


def test_l4s_archive_requires_matching_image_and_binary_mask(tmp_path):
    import io
    import zipfile
    import h5py
    import numpy as np
    from risk.r4_labels import inspect_l4s_archive
    def h5(key, array):
        stream = io.BytesIO()
        with h5py.File(stream, 'w') as f:
            f[key] = array
        return stream.getvalue()
    archive = tmp_path / 'training.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('TrainData/img/image_1.h5', h5('img', np.zeros((128, 128, 14))))
        z.writestr('TrainData/mask/mask_1.h5', h5('mask', np.ones((128, 128), dtype='uint8')))
    result = inspect_l4s_archive(archive, 40000000)
    assert result['image_shape'] == [128, 128, 14]
    assert result['mask_values'] == [1]
    assert result['labelled_pair_verified']
