"""Pipeline-level tests for F5/A8 on synthetic fixtures: no OpenSR-test data and no model are needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments import f5_a8_mapping as f5
from trustsr import alloc, hrbench as hb

ROOT = Path(__file__).resolve().parents[1]
SPECS = [('nonveg_ndvi_0.35', 'ndvi', 0.35, (0, 3))]
TOL = (1, 2)


def synthetic_scene(size_lr=8, seed=0):
    """An HR scene of vegetation with bare patches, its exact 10 m block mean, and a perfect 'SR' = the HR itself."""
    rng = np.random.default_rng(seed)
    h = size_lr * f5.BLOCK
    veg = np.array([0.03, 0.05, 0.03, 0.40])
    bare = np.array([0.30, 0.25, 0.20, 0.25])
    is_bare = np.zeros((h, h), bool)
    for _ in range(12):
        r, c = rng.integers(0, h - 6, 2)
        is_bare[r:r + rng.integers(1, 6), c:c + rng.integers(1, 6)] = True
    hr = np.where(is_bare[None], bare[:, None, None], veg[:, None, None]).astype(np.float64)
    hr += 0.002 * rng.standard_normal(hr.shape)
    lr = hb.block_mean(hr, f5.BLOCK)
    return lr, hr, is_bare


def test_class_specs_are_tied_to_the_preregistered_config():
    a8 = yaml.safe_load((ROOT / 'configs' / 'fix.yaml').read_text(encoding='utf-8'))['a8']
    specs = f5.class_specs(a8)
    assert [s[0] for s in specs] == ['nonveg_ndvi_0.35', 'nonveg_ndvi_0.25', 'nonveg_ndvi_0.45', 'water_ndwi_0.0']
    assert specs[0][2] == 0.35, 'the primary class must be the pre-registered 0.35 threshold'
    drifted = {'truth': {'primary': 'HR NDVI < 0.99 -> non-vegetation', 'secondary': 'NDWI > 0 -> water'}}
    with pytest.raises(ValueError):
        f5.class_specs(drifted)


def test_score_sign_convention_makes_low_score_the_target():
    refl = np.zeros((4, 1, 1))
    refl[:, 0, 0] = [0.3, 0.25, 0.2, 0.25]                      # bare: low NDVI
    assert f5.score_from(refl, 'ndvi')[0, 0] < f5.score_threshold('ndvi', 0.35)
    water = np.zeros((4, 1, 1))
    water[:, 0, 0] = [0.02, 0.06, 0.05, 0.02]                   # NDWI > 0
    assert f5.score_from(water, 'ndwi')[0, 0] < f5.score_threshold('ndwi', 0.0)


def test_evaluate_image_detects_a_perfect_sr_beating_bilinear():
    """With SR = the HR image itself, the SR ranking is perfect and must beat the bilinear ranking on real IoU.

    This is the end-to-end sanity check that the harness can see a ranking difference when one genuinely exists.
    """
    lr, hr, _ = synthetic_scene()
    valid = np.ones(hr.shape[-2:], bool)
    out = f5.evaluate_image(lr, hr, valid, hr, SPECS, TOL)
    ms = out['classes']['nonveg_ndvi_0.35']['methods']
    assert ms['alloc_pretrained_sr|oracle']['iou'] == pytest.approx(1.0)
    assert ms['alloc_bilinear|oracle']['iou'] < 1.0
    assert ms['alloc_pretrained_sr|oracle']['iou'] > ms['alloc_bilinear|oracle']['iou']
    assert ms['alloc_pretrained_sr|oracle']['boundary_f1']['1']['f1'] == pytest.approx(1.0)
    assert ms['alloc_pretrained_sr|oracle']['area_error']['relative'] == pytest.approx(0.0)


def test_evaluate_image_oracle_fraction_is_never_worse_than_realistic_on_a_clean_fixture():
    lr, hr, _ = synthetic_scene(seed=4)
    valid = np.ones(hr.shape[-2:], bool)
    ms = f5.evaluate_image(lr, hr, valid, hr, SPECS, TOL)['classes']['nonveg_ndvi_0.35']['methods']
    for m in f5.ALLOC_METHODS:
        assert ms[f'{m}|oracle']['iou'] >= ms[f'{m}|realistic']['iou']


def test_evaluate_image_reports_every_method_fraction_and_stratum():
    lr, hr, _ = synthetic_scene(seed=2)
    valid = np.ones(hr.shape[-2:], bool)
    ms = f5.evaluate_image(lr, hr, valid, hr, SPECS, TOL)['classes']['nonveg_ndvi_0.35']['methods']
    assert set(ms) == {'blocky_10m', 'hard_threshold_v1_style',
                       'alloc_bilinear|oracle', 'alloc_bilinear|realistic',
                       'alloc_pretrained_sr|oracle', 'alloc_pretrained_sr|realistic'}
    for rec in ms.values():
        assert set(rec['boundary_f1']) == {'1', '2'}
        assert set(rec['strata']) == set(alloc.WIDTH_BIN_NAMES)


def test_blocky_never_has_subpixel_structure_on_a_real_shaped_fixture():
    lr, hr, _ = synthetic_scene(seed=7)
    s10 = f5.score_from(lr, 'ndvi')
    blocky = alloc.blocky_mask(s10, 0.35, f5.BLOCK)
    per_block = alloc._blocks(blocky, f5.BLOCK)
    assert np.all(per_block.all(axis=-1) | (~per_block).all(axis=-1))


def test_paired_delta_and_summary_denominators():
    lr, hr, _ = synthetic_scene(seed=5)
    valid = np.ones(hr.shape[-2:], bool)
    records = [{'dataset': ds, 'name': f'roi{i}', 'group': 'g', 'valid_px': int(valid.sum()),
                'metrics': f5.evaluate_image(lr, hr, valid, hr, SPECS, TOL)}
               for i, ds in enumerate(['naip', 'spot', 'naip'])]
    boot = {'replicates': 200, 'ci': 0.95, 'seed': 1}
    d = f5.paired_delta(records, 'nonveg_ndvi_0.35', 'alloc_pretrained_sr|oracle', 'alloc_bilinear|oracle', boot)
    assert d['n_images'] == 3 and d['n'] == 3
    assert d['lo'] <= d['mean'] <= d['hi']
    no_naip = f5.paired_delta(records, 'nonveg_ndvi_0.35', 'alloc_pretrained_sr|oracle', 'alloc_bilinear|oracle',
                              boot, subset=lambda r: r['dataset'] != 'naip')
    assert no_naip['n_images'] == 1
    s = f5.summarize(records, TOL, 'nonveg_ndvi_0.35')
    assert s['ALL']['n_images'] == 3 and s['ALL_no_naip']['n_images'] == 1
    assert s['naip']['n_images'] == 2
    entry = s['ALL']['methods']['alloc_pretrained_sr|oracle']
    assert entry['n_images_iou_defined'] == 3
    assert entry['pooled_area_error']['truth_px'] > 0


def test_finetuned_and_heavy_sr_are_not_evaluated_anywhere():
    """alloc_finetuned_sr must stay BLOCKED: it must never appear as an evaluated method."""
    lr, hr, _ = synthetic_scene(seed=9)
    valid = np.ones(hr.shape[-2:], bool)
    ms = f5.evaluate_image(lr, hr, valid, hr, SPECS, TOL)['classes']['nonveg_ndvi_0.35']['methods']
    assert not any('finetuned' in k or 'heavy' in k for k in ms)
    assert f5.ALLOC_METHODS == ('alloc_bilinear', 'alloc_pretrained_sr')
