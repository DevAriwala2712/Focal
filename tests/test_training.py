import json

import numpy as np
import pytest


def prepared_pair(tmp_path, *, shift=0):
    from rasterio.transform import from_origin
    from risk.calibration import write_cog
    from risk.common import digest
    bands = ['B04', 'B03', 'B02', 'B08']
    write_cog(tmp_path / 'lr.tif', np.ones((4, 2, 2), dtype='float32'), 'EPSG:32643',
              from_origin(500000, 1300000, 10, 10), bands, None)
    write_cog(tmp_path / 'hr.tif', np.ones((4, 8, 8), dtype='float32'), 'EPSG:32643',
              from_origin(500000 + shift, 1300000, 2.5, 2.5), bands, None)
    return {'dataset': 'WorldStrat', 'pairs': [dict(aoi_id='fixture', lr='lr.tif', hr='hr.tif',
        lr_sha256=digest(tmp_path / 'lr.tif'), hr_sha256=digest(tmp_path / 'hr.tif'),
        source_urls=['https://example.test/synthetic-unit-fixture'], split='train',
        lr_scale=1, lr_offset=0, hr_scale=1, hr_offset=0)]}


def test_pair_grid_shift_is_rejected(tmp_path):
    from risk.r5_finetune import validate_manifest
    manifest = prepared_pair(tmp_path, shift=1)
    with pytest.raises(ValueError, match='bounds'):
        validate_manifest(manifest, tmp_path, ['B04','B03','B02','B08'])


def test_zero_radiometric_scale_is_rejected(tmp_path):
    from risk.r5_finetune import validate_manifest
    manifest = prepared_pair(tmp_path)
    manifest['pairs'][0]['lr_scale'] = 0
    with pytest.raises(ValueError, match='scale'):
        validate_manifest(manifest, tmp_path, ['B04','B03','B02','B08'])


def test_cpu_training_updates_weights_for_all_fifty_steps():
    import torch
    from risk.r5_finetune import train_steps
    # Tiny artificial model tests the optimizer loop only, never the R5 science claim.
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    samples = [(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))]
    losses, delta, grads = train_steps(model, samples, 50, 0.01, 'cpu')
    assert len(losses) == 50
    assert losses[-1] < losses[0]
    assert delta > 0
    assert grads
