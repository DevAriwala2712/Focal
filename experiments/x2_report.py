"""Markdown report for x2.json (numbers only come from the result dict)."""
from __future__ import annotations

from pathlib import Path


def _f(x, n=3):
    return 'n/a' if x is None else f'{x:.{n}f}'


def write_report(res: dict, path: Path) -> None:
    L = [f'# A2: HR validation benchmark (x2)\n', f'Status: **{res["status"]}**  |  evidence: **{res["evidence"]}**  |  verdict: **{res.get("verdict", "n/a")}**\n']
    if res['status'] == 'BLOCKED' or 'datasets' not in res:
        L.append(f'\nReason: {res.get("reason") or res.get("error")}\n')
        Path(path).write_text('\n'.join(L), encoding='utf-8')
        return
    p = res['protocol']
    L.append(f'\n{res["summary"]}. Model: pinned SEN2SR-lite RGBN x4 (B04,B03,B02,B08), device `{res["device"]}`. '
             f'Reference: OpenSR-test HRharm, reflectance = DN/10000, data range {p["data_range"]}, {p["border_px"]} px border excluded, '
             f'candidates clipped to [0,1]. CI = paired bootstrap over images ({p["bootstrap"]["replicates"]} replicates, '
             f'{int(p["bootstrap"]["ci"] * 100)} %, seed {p["bootstrap"]["seed"]}). Keep rule: PSNR CI lower bound > 0.\n')
    L.append('\n| dataset | n images (valid HR px) | metric | nearest | bicubic | lanczos | SR | delta vs bicubic | 95 % CI | evidence |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for ds, d in res['datasets'].items():
        for k, unit in (('psnr', 'PSNR dB'), ('ssim', 'SSIM')):
            m, dv = d['mean'][k], d['delta_vs_bicubic'][k]
            nd = 3 if k == 'psnr' else 4
            L.append(f'| {ds} | {d["n_images"]} ({d["n_valid_px_hr"]}) | {unit} | {_f(m["nearest"], nd)} | {_f(m["bicubic"], nd)} | '
                     f'{_f(m["lanczos"], nd)} | {_f(m["sr"], nd)} | {dv["mean"]:+.{nd}f} | [{dv["lo"]:+.{nd}f}, {dv["hi"]:+.{nd}f}] | real |')
    L.append('\n## Per-dataset verdict and strongest baseline\n')
    L.append('| dataset | SR beats bicubic | strongest baseline | SR minus strongest (PSNR dB, CI) | images SR>bicubic | by-S2-acquisition CI (n groups) |')
    L.append('|---|---|---|---|---|---|')
    for ds, d in res['datasets'].items():
        s, g = d['delta_vs_strongest']['psnr'], d['delta_vs_bicubic_by_group']['psnr']
        L.append(f'| {ds} | {"YES" if d["beats_bicubic"] else "NO"} | {d["strongest_baseline"]} | {s["mean"]:+.3f} [{s["lo"]:+.3f}, {s["hi"]:+.3f}] | '
                 f'{d["images_sr_better_than_bicubic_psnr"]}/{d["n_images"]} | {g["mean"]:+.3f} [{g["lo"]:+.3f}, {g["hi"]:+.3f}] ({d["n_groups"]}) |')
    pl = res['pooled']
    pc = pl['delta_psnr_all_images_ci']
    L.append(f'\nPooled ({pl["n_images"]} images, {pl["n_datasets"]} datasets): mean PSNR delta {pl["mean_delta_psnr_all_images"]:+.3f} dB '
             f'[{pc["lo"]:+.3f}, {pc["hi"]:+.3f}]; mean of dataset means {pl["mean_of_dataset_mean_delta_psnr"]:+.3f} dB.\n')
    L.append('\n## 10 m downsample error (block mean 4x4 of unclipped SR vs the 10 m input; mean over images, reflectance units, bands B04,B03,B02,B08)\n')
    L.append('| dataset | method | MAE per band | bias per band |')
    L.append('|---|---|---|---|')
    for ds, d in res['datasets'].items():
        for m in ('sr', 'bicubic', 'lanczos'):
            e = d['mean_downsample_error'][m]
            L.append(f'| {ds} | {m} | {", ".join(f"{v:.5f}" for v in e["mae"])} | {", ".join(f"{v:+.5f}" for v in e["bias"])} |')
    if res.get('opensr_test_groups'):
        L.append(f'\n## opensr-test {res["data"]["opensr_test_version"]} groups (mean over images; consistency: reflectance, spectral, spatial; synthesis; correctness: ha/om/im percent)\n')
        L.append('| dataset | method | n ok | reflectance | spectral (deg) | spatial | synthesis | hallucination | omission | improvement |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        for ds, ms in res['opensr_test_groups'].items():
            for m, v in ms.items():
                L.append(f'| {ds} | {m} | {v["n_ok"]}/{v["n_ok"] + v["n_failed"]} | {_f(v["reflectance"], 5)} | {_f(v["spectral"], 4)} | {_f(v["spatial"], 4)} | '
                         f'{_f(v["synthesis"], 5)} | {_f(v["ha_metric"], 4)} | {_f(v["om_metric"], 4)} | {_f(v["im_metric"], 4)} |')
    L.append('\n## Sensitivity: unharmonized HR (only sets whose raw radiometry is on the S2 scale)\n')
    L.append('| dataset | n | SR PSNR | bicubic PSNR | delta | 95 % CI |')
    L.append('|---|---|---|---|---|---|')
    for ds, d in res['sensitivity_raw_hr']['datasets'].items():
        dv = d['delta_vs_bicubic']['psnr']
        L.append(f'| {ds} | {d["n_images"]} | {d["mean"]["psnr"]["sr"]:.3f} | {d["mean"]["psnr"]["bicubic"]:.3f} | {dv["mean"]:+.3f} | [{dv["lo"]:+.3f}, {dv["hi"]:+.3f}] |')
    L.append('\n## Geometry check (does HRharm sit on the exact LR subdivision?)\n')
    L.append('| dataset | scale | HR origin on 10 m grid | shift search within 1 HR px | dataset-reported misalignment (LR px, mean/max) | S2 acquisitions |')
    L.append('|---|---|---|---|---|---|')
    for ds, g in res['geometry'].items():
        L.append(f'| {ds} | x{g["ref_scale"]} | {g["hr_origin_on_10m_grid"]}/{g["n"]} | {g["shift_search_within_1_hr_px"]}/{g["n"]} | '
                 f'{g["meta_spatial_misalignment_lr_px"]["mean"]:.3f} / {g["meta_spatial_misalignment_lr_px"]["max"]:.3f} | {g["unique_s2_acquisitions"]} |')
    dt = res['determinism']
    L.append(f'\n## Reproducibility\n\nSeed {dt["seed"]}, deterministic algorithms on, torch threads {dt["torch_threads"]}, device {res["device"]}. '
             f'Full SR recompute gives identical sha256 for every dataset: {dt["full_recompute_hashes_equal"]}. '
             f'Bytes downloaded by this process: {res["data"]["bytes_downloaded_this_process"]} (cap {res["data"]["byte_cap"]}); dataset revision {res["data"]["revision"]}.\n')
    L.append('\n## Limitations\n')
    L += [f'- {x}' for x in res['limitations']]
    Path(path).write_text('\n'.join(L) + '\n', encoding='utf-8')
