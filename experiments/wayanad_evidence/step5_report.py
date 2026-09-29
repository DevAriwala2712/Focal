"""Step 5: figure.png, EVIDENCE.md and results.json, generated from the step JSONs (no number is typed by hand)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from experiments.common import environment_record, finalize_result
from experiments.wayanad_evidence import config as C
from experiments.wayanad_evidence.data import load_date
from experiments.wayanad_evidence.gate import INFERRED, NO_DATA, OBSERVED, UNSUPPORTED
from risk.common import write_json


def _rgb(refl, gain):
    return np.clip(np.moveaxis(refl[:3], 0, -1) * gain, 0, 1)


def make_figure(cfg, root, out, cache, s1, s4, labels_text, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from rasterio import open as ropen
    r0, r1, c0, c1 = (int(v) for v in s1['crop'])
    pre, post = [str(d) for d in s1['pre']], str(s1['post'])
    gain = cfg['figure']['rgb_gain']
    pre_refl = []
    for d in pre:
        a = load_date(cfg, cache, d)
        r = a['refl'][:, r0:r1, c0:c1].astype('float64')
        r[:, ~a['valid_scl'][r0:r1, c0:c1]] = np.nan
        pre_refl.append(r)
    with np.errstate(all='ignore'):
        pre_mean = np.nanmean(pre_refl, axis=0)
    apost = load_date(cfg, cache, post)
    post_refl = apost['refl'][:, r0:r1, c0:c1]
    sr_post = np.load(cache / f'sr_identity_{post}.npy')
    with ropen(out / 'class_map_2p5m.tif') as src:
        cls = src.read(1)
    h, w = cls.shape
    ext10 = (0, (c1 - c0) * 10, (r1 - r0) * 10, 0)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 13), dpi=cfg['figure']['dpi'])
    fig.suptitle(labels_text['title'], fontsize=12, y=0.995)
    axes[0, 0].imshow(np.nan_to_num(_rgb(pre_mean, gain), nan=0.5), extent=ext10)
    axes[0, 0].set_title(f"10 m pre RGB: pixel-wise mean of {', '.join(pre)} (valid pixels; grey = cloud/shadow on all)", fontsize=9)
    axes[0, 1].imshow(_rgb(post_refl, gain), extent=ext10)
    axes[0, 1].set_title(f'10 m post RGB: {post} (acquired {labels_text["gap_days"]} days after the event)', fontsize=9)
    axes[1, 0].imshow(_rgb(sr_post, gain), extent=ext10)
    axes[1, 0].set_title(f'2.5 m SR post RGB, {post}: model reconstruction\n(pretrained SEN2SR-lite, NOT fine-tuned); '
                         'not a new observation', fontsize=9)
    base = _rgb(sr_post, gain)
    grey = base.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.85
    overlay = grey.copy()
    for code, colour in ((OBSERVED, (0.86, 0.1, 0.1)), (INFERRED, (1.0, 0.72, 0.0)), (NO_DATA, (0.55, 0.55, 0.55))):
        m = cls == code
        overlay[m] = 0.35 * grey[m] + 0.65 * np.array(colour)
    axes[1, 1].imshow(overlay, extent=ext10)
    axes[1, 1].set_title(f'2.5 m trust-gate classes (k = 2.0, NOT calibrated; parent threshold provisional)', fontsize=9)
    m = s4['class_counts_px']
    ar = s4['class_areas_m2']
    axes[1, 1].legend(handles=[
        Patch(color=(0.86, 0.1, 0.1), label=f"OBSERVED: {m['OBSERVED']:,} px ({ar['OBSERVED']:,.0f} m²)"),
        Patch(color=(1.0, 0.72, 0.0), label=f"INFERRED: {m['INFERRED']:,} px ({ar['INFERRED']:,.0f} m²)"),
        Patch(color=(0.55, 0.55, 0.55), label=f"NO_DATA: {m['NO_DATA']:,} px"),
        Patch(facecolor='white', edgecolor='k', label=f"UNSUPPORTED (excluded, not drawn): {m['UNSUPPORTED']:,} px")],
        loc='lower left', fontsize=8, framealpha=0.92)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.plot([150, 650], [(r1 - r0) * 10 - 100] * 2, color='w', lw=3)
        ax.text(400, (r1 - r0) * 10 - 160, '500 m', color='w', ha='center', fontsize=8)
    fig.text(0.5, 0.005, labels_text['footer'], ha='center', fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(path)
    plt.close(fig)


def table_row(name, value, unit, denominator, source):
    return f'| {name} | {value} | {unit} | {denominator} | `{source}` |'


def build(cfg, root, config_hash):
    out, cache = C.outputs(cfg, root), C.cache(cfg, root)
    step = {n: json.loads((out / f'{n}.json').read_text(encoding='utf-8')) for n in ('step1', 'step2', 'step3', 'step4')
            if (out / f'{n}.json').is_file()}
    statuses = {n: s['status'] for n, s in step.items()}
    missing = [n for n in ('step1', 'step2', 'step3', 'step4') if n not in step]
    if any(statuses.get(n) != 'PASS' for n in ('step1', 'step2', 'step3', 'step4')):
        raise RuntimeError(f'cannot build the evidence pack: step statuses {statuses}, missing {missing}')
    m1, m2, m3, m4 = (step[n]['measurements'] for n in ('step1', 'step2', 'step3', 'step4'))
    with np.load(cache / 'step1_state.npz') as z:
        s1 = {k: z[k] for k in z.files}
    dates = m1['dates']
    labels_text = {
        'title': (f"Wayanad landslide, retrospective NDVI-change evidence: pre {', '.join(dates['pre'])}, post {dates['post']} "
                  f"({dates['gap_days_event_to_post']} days after 2024-07-30)\npretrained SEN2SR-lite (NOT fine-tuned) - "
                  f"k = 2.0 (NOT calibrated) - retrospective comparison"),
        'gap_days': dates['gap_days_event_to_post'],
        'footer': (f"Post-event image is {dates['gap_days_event_to_post']} days after the event: retrospective vegetation-disturbance "
                   "comparison, not rapid response. NDVI drop is not uniquely landslide damage. No accuracy or F1 is claimed.")}
    make_figure(cfg, root, out, cache, s1, m4, labels_text, out / 'figure.png')

    aoi_area = float(np.prod(m1['aoi']['shape_px']) * 100)
    c = m4['class_counts_px']
    a = m4['class_areas_m2']
    rho = m4['rho']
    rows = [
        table_row('10 m parent area, AOI', f"{m1['parent']['area_m2_aoi']:,.0f}", 'm²', f'{aoi_area:,.0f} m² AOI (1024x1024 px @10 m)', 'outputs/step1.json'),
        table_row('10 m parent area, largest component', f"{m1['parent']['largest_component_area_m2']:,.0f}", 'm²', 'parent-positive pixels, 8-connected', 'outputs/step1.json'),
        table_row('10 m parent area, inside 2.5 m crop', f"{m4['baselines']['parent_area_10m_crop_m2']:,.0f}", 'm²', f"crop {m1['crop']['shape_px']} px @10 m", 'outputs/step4.json'),
        table_row('Footprint (largest component + buffer)', f"{m1['footprint']['area_m2']:,.0f}", 'm²', f"buffer {m1['footprint']['buffer_m']} m; date selection only", 'outputs/step1.json'),
        table_row('OBSERVED', f"{c['OBSERVED']:,} px = {a['OBSERVED']:,.0f}", 'px = m²', f"{m4['valid_px']:,} valid 2.5 m px", 'outputs/step4.json'),
        table_row('INFERRED', f"{c['INFERRED']:,} px = {a['INFERRED']:,.0f}", 'px = m²', f"{m4['valid_px']:,} valid 2.5 m px", 'outputs/step4.json'),
        table_row('OBSERVED ∪ INFERRED', f"{m4['observed_union_inferred_area_m2']:,.0f}", 'm²', 'equals the parent area on valid pixels', 'outputs/step4.json'),
        table_row('UNSUPPORTED (SR-only, suppressed by the gate)', f"{c['UNSUPPORTED']:,}", 'px', f"of {m4['S_ungated_positive_px']:,} ungated S pixels", 'outputs/step4.json'),
        table_row('Ungated SR positives S = d > kσ', f"{m4['S_ungated_positive_px']:,} px = {m4['S_ungated_area_m2']:,.0f}", 'px = m²', f"{m4['valid_px']:,} valid 2.5 m px", 'outputs/step4.json'),
        table_row('NO_DATA (cloud/shadow/invalid/NDVI undefined)', f"{c['NO_DATA']:,}", 'px', f"{np.prod(m4['crop_2p5m_px']):,} crop px @2.5 m", 'outputs/step4.json'),
        table_row('NRSC main scarp (scale check ONLY, not a label)', f"{m4['nrsc_scale_check']['nrsc_main_scarp_m2']:,}", 'm²', 'different definition and data', 'configs/wayanad_evidence.yaml'),
        table_row('ρ median, S-objects ≥ 16 px', f"{rho['objects_ge_min_px_distribution'].get('median', float('nan')):.3f}", 'fraction of energy', f"{rho['objects_ge_min_px']} objects", 'outputs/step4.json'),
        table_row('ρ p10 / p90, S-objects ≥ 16 px', f"{rho['objects_ge_min_px_distribution'].get('p10', float('nan')):.3f} / {rho['objects_ge_min_px_distribution'].get('p90', float('nan')):.3f}", 'fraction of energy', f"{rho['objects_ge_min_px']} objects", 'outputs/step4.json'),
        table_row('ρ energy-weighted mean, all S-objects', f"{rho['energy_weighted_mean_rho']:.3f}" if rho['energy_weighted_mean_rho'] is not None else 'n/a', 'fraction of energy', f"{rho['objects_total']} objects", 'outputs/step4.json'),
        table_row('ρ median, majority-UNSUPPORTED objects', f"{rho['majority_unsupported_objects'].get('median', float('nan')):.3f}", 'fraction of energy', f"{rho['majority_unsupported_objects'].get('n', 0)} objects", 'outputs/step4.json'),
        table_row('Post-event gap', f"{dates['gap_days_event_to_post']}", 'days', 'event 2024-07-30 to post date', 'outputs/step1.json'),
        table_row('Total SR wall time (8 runs, all tiles, 4 dates)', f"{m3['sr']['seconds_per_tile_8_runs']['total']:.0f}", 's', f"{m3['sr']['forward_passes_total']} forward passes on {m3['sr']['device']}", 'outputs/step3.json'),
        table_row('Median time per tile (8 dihedral runs)', f"{m3['sr']['seconds_per_tile_8_runs']['median']:.2f}", 's', f"{m3['sr']['tiles_per_date']} tiles per date", 'outputs/step3_tile_log.json'),
        table_row('Peak process RSS (CPU proxy)', f"{m3['sr']['memory'].get('peak_rss_mib', float('nan')):.0f}", 'MiB', 'NOT GPU memory', 'outputs/step3.json'),
        table_row('GPU peak allocated/reserved', 'BLOCKED', '-', 'no CUDA device on the run machine', 'outputs/step3.json'),
    ]
    sc_rows = []
    for d_, v in m4['spectral_consistency'].items():
        b = v['per_band_mae_identity_run']
        sc_rows.append(f"| {d_} | " + ' | '.join(f"{b[k]:.4f}" for k in ('B04', 'B03', 'B02', 'B08')) + f" | {v['valid_10m_px']:,} |")
    s2rows = [f"| {t['date']} | {t['days_after_event']} | {t['aoi_cloud_shadow_pct']:.1f} | {100 * t['footprint_valid_fraction']:.1f} | "
              f"{t['footprint_valid_px']:,} / {t['footprint_px']:,} | {'yes' if t['passes'] else 'no'} |" for t in m2['table']]
    e = m2['earliest_passing_date']
    mask = m3['mask_check']
    seam = mask['real_model_seam_probe']['by_distance_from_tile_edge']
    seam_rows = ' | '.join(f"{k}: p99 {v['p99_abs_delta_between_grids']:.4g}" for k, v in seam.items())
    replaced = dates['replaced_vs_preferred']
    replaced_text = (f"Dates differ from the brief where its dates failed the AOI clear rule on this AOI: {replaced}." if replaced
                     else "The brief's preferred dates passed the clear rule on this AOI.")
    earliest_text = (f"{e['date']} ({e['days_after_event']} days after the event, {100 * e['footprint_valid_fraction']:.1f} % valid)"
                     if e else 'none passes the threshold')
    distances_text = ', '.join(f'{k} {v:.0f} m' for k, v in m1['plausibility']['distance_m_to_published_points'].items())
    text = f"""# Wayanad evidence: NDVI change with a trust gate

**Labels (apply to every number and image here):** {cfg['labels']['model']} · {cfg['labels']['k']} · {cfg['labels']['comparison']}.
Pre-event acquisitions {', '.join(dates['pre'])}; post-event acquisition **{dates['post']}**, **{dates['gap_days_event_to_post']} days** after the 2024-07-30 event.
Parent-drop threshold {m1['parent']['threshold']}: **{m1['parent']['threshold_status']}**, not calibrated. Bands B02/B03/B04/B08 only; SCL is a validity mask only.

**AOI correction.** The AOI in `RISK_REPORT.md` (11.782N, 76.233E) is about 34 km north of the slide. This run uses a 10,240 m square centred on
{cfg['aoi']['latitude']}N, {cfg['aoi']['longitude']}E (chosen from published place coordinates, never from imagery), so every date and item ID differs from
`risk/results/r2_acquisitions.json`. The largest parent-positive component lies within {m1['plausibility']['radius_m']} m of every published coordinate:
{distances_text}.
{replaced_text}

![figure](figure.png)

## Numbers

| Quantity | Value | Unit | Denominator / basis | Source |
|---|---|---|---|---|
""" + '\n'.join(rows) + f"""

The NRSC figure sits beside these only for scale. TrustSR areas are NDVI-drop pixels (scarp plus runout vegetation loss, on a
10 m parent rule); NRSC maps the main scarp from other data with another definition. NRSC is not a label and no agreement is claimed.

ρ = ‖PΔ‖²/‖Δ‖² per connected S-object (P = 4×4 block mean): a fraction of signal energy, not a probability. Objects smaller than one 10 m block
(16 px) are excluded from the main distribution; all {rho['objects_total']} objects are in `outputs/rho_objects.csv`.

### Spectral consistency (mean |area-mean 4×4 downsample of SR − 10 m input| reflectance, identity run, valid pixels)

| Date | B04 | B03 | B02 | B08 | 10 m valid px |
|---|---|---|---|---|---|
""" + '\n'.join(sc_rows) + f"""

Reported against a tolerance of {m4['spectral_tolerance_mae_reported_against']} (set in config, not tuned). The pretrained constraint mixes the *bicubic*-upsampled
input into low frequencies, so a nonzero downsample error is expected and is not a fidelity score.

## Step 2: was the scar visible before {dates['post']}? (ADR-002, report only)

Candidates: every acquisition after the event and before the chosen post date with AOI cloud+shadow < 100 % on this AOI. Threshold: ≥ {100 * m2['threshold_min_valid_fraction']:.0f} % of footprint pixels SCL-valid.

| Date | Days after event | AOI cloud+shadow % | Footprint valid % | Valid / footprint px | Passes |
|---|---|---|---|---|---|
""" + '\n'.join(s2rows) + f"""

Earliest passing date: **{earliest_text}**.
The post date of this run was not switched. The footprint is used for date selection only, never as a label; SCL validity is a model estimate and does not mean haze-free.

## Step 3a: the pinned Fourier mask

Shape {mask['shape']}, {mask['dtype']}, {mask['unique_values']} unique values (min {mask['min']}, max {mask['max']}); centre {mask['centre_value']}, corner {mask['corner_value']}.
Best-matching upstream family at radius {mask['radius_used_for_fit']}: **{mask['fit']['best_family']}** (max |diff| {mask['fit']['fits'][mask['fit']['best_family']]['max_abs_diff']:.3g}; reproduces within 1e-6: {mask['fit']['reproduces_within_1e-6']}); binary: {mask['fit']['is_binary']} → classified **{mask['classification']}**.
Stride used: {mask['stride_used']} with a {cfg['tiling']['crop_margin_px']} px centre-crop margin. {mask['stride_note']}.
Real-model seam probe (two tile grids offset by 64 input px, real weights, real imagery; p99 |Δ| by 2.5 m px from the tile edge): {seam_rows}.
Synthetic probe for comparison: `{mask['synthetic_edge_probe']['source']}`. The two are different inputs; compare shape of decay, not absolute values.
Upstream `gaussian_filter` and `sigmoid_filter` raise `TypeError` in the pinned package, so the families were re-implemented from their source formulas (test-checked).

## Runtime and memory

Device **{m3['sr']['device']}** (CUDA available: {m3['sr']['cuda_available']}). {m3['sr']['tiles_per_date']} native 128 px tiles per date at stride {cfg['tiling']['stride']}, 8 dihedral runs each, {len(dates['pre']) + 1} dates
= {m3['sr']['forward_passes_total']} forward passes; median {m3['sr']['seconds_per_tile_8_runs']['median']:.2f} s per tile (8 runs), total {m3['sr']['seconds_per_tile_8_runs']['total']:.0f} s.
Peak process RSS {m3['sr']['memory'].get('peak_rss_mib', float('nan')):.0f} MiB (CPU proxy). **GPU peak memory and GPU latency: BLOCKED**, no CUDA device here; the only GPU evidence is R1 (synthetic input, RTX 4050).
Determinism: seed {cfg['seed']}, `torch.use_deterministic_algorithms(True)`, fixed tile order; a tile re-run was byte-identical: {m3['sr']['determinism']['tile_rerun_bytes_identical']} (macOS-arm64 CPU only).
Network: {m1['bytes_received']['bytes_received']:,} bytes received (cap {m1['bytes_received']['cap_bytes']:,}).

## What this does not show

- **No accuracy, precision, recall or F1.** No labels were used or compared against; NRSC is a scale check only.
- **No calibrated k and no calibrated parent threshold.** k = 2.0 and the threshold {m1['parent']['threshold']} are configured defaults fixed before viewing data.
- **No fine-tuning.** Weights are the pretrained SEN2SR-lite; R5 (fine-tune) has not run.
- **No 2.5 m ground truth.** SR output is a model reconstruction; UNSUPPORTED/INFERRED describe measurement support, not truth.
- **No physical 4 GB test.** No CUDA device was available; no GPU memory number exists for this run.
- **No building- or road-damage claim.** NDVI drop is vegetation disturbance, not uniquely landslide damage.
- **Seasonal/recovery confounding.** Dry-season (Jan-Feb) pre dates against a {dates['post']} post date; the post-monsoon vegetation is not the same season.
- **Reflectance scaling.** ESA baseline convention (DN−1000)/10000 was applied; the SEN2SR training-data scaling was not verified against the model card.
- **Circular footprint.** The footprint comes from the same NDVI comparison and is used only for date selection and cropping.

## Reproduce

```bash
.venv/bin/python -m experiments.wayanad_evidence.step1_baseline   # audit, fetch, baseline (network)
.venv/bin/python -m experiments.wayanad_evidence.step2_visibility
.venv/bin/python -m experiments.wayanad_evidence.step3_sr
.venv/bin/python -m experiments.wayanad_evidence.step4_gate
.venv/bin/python -m experiments.wayanad_evidence.step5_report
```
Config hash `{config_hash}`. Large COGs (`stack_*.tif`, `sr_*.tif`) are regenerated locally and not committed.
"""
    (root / cfg['paths']['results'] / 'EVIDENCE.md').write_text(text, encoding='utf-8')

    started = datetime.now(timezone.utc).isoformat()
    result = {'status': 'PASS', 'evidence': 'real',
              'measurements': {'labels': cfg['labels'], 'dates': dates, 'step_status': statuses,
                               'parent': m1['parent'], 'footprint': m1['footprint'], 'crop': m1['crop'],
                               'class_counts_px': c, 'class_areas_m2': a, 'ungated_S_px': m4['S_ungated_positive_px'],
                               'unsupported_suppressed_px': c['UNSUPPORTED'], 'rho': {k: rho[k] for k in (
                                   'objects_total', 'objects_ge_min_px', 'all_objects', 'objects_ge_min_px_distribution',
                                   'energy_weighted_mean_rho', 'majority_unsupported_objects')},
                               'spectral_consistency': m4['spectral_consistency'],
                               'step2_earliest_passing_date': m2['earliest_passing_date'],
                               'runtime': m3['sr']['seconds_per_tile_8_runs'], 'device': m3['sr']['device']},
              'blocked_items': [{'item': 'GPU peak memory / GPU latency / physical 4 GB test',
                                 'missing': 'a CUDA device (the RTX 4050 machine); this run used the macOS CPU'}],
              'limitations': sorted({lim for s in step.values() for lim in s.get('limitations', [])})}
    final = finalize_result('wayanad_evidence', result, config_hash, started)
    final['environment'] = environment_record()
    write_json(root / cfg['paths']['results'] / 'results.json', final)
    return final


def main():
    cfg, root, config_hash = C.load()
    final = build(cfg, root, config_hash)
    print(json.dumps({k: final[k] for k in ('status', 'evidence', 'experiment', 'finished_utc')}, indent=2))


if __name__ == '__main__':
    main()
