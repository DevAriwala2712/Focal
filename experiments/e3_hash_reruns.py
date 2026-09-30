"""E3: hash-checked reruns. Seeded, deterministic-algorithm E1 runs; SHA-256 the output arrays."""
from __future__ import annotations

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # must precede CUDA init for deterministic cuBLAS

import argparse
import json
import random
import subprocess
import sys

import numpy as np

from experiments.common import Blocked, array_sha256, run_cli, synthetic_scene
from experiments.e1_tile_scheduler import model_operator, super_resolve_tiled, tile_origins


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare_runs(arrays: list) -> dict:
    if len(arrays) < 2:
        raise ValueError('need at least two runs to compare')
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError(f'run shape mismatch: {[a.shape for a in arrays]}')
    hashes = [array_sha256(a) for a in arrays]
    delta = max(float(np.abs(a.astype(np.float64) - arrays[0].astype(np.float64)).max()) for a in arrays[1:])
    return {'sha256': hashes, 'identical': len(set(hashes)) == 1, 'max_abs_delta': delta}


def run_e1(cfg, root, device: str, threads: int | None, reverse_tiles: bool = False) -> np.ndarray:
    """Seeded E1 scheduler run with the real pinned model. Tile order is fixed (row-major)."""
    import torch
    from risk.model import load_model
    s1, s3, p0 = cfg['e1'], cfg['e3'], cfg['phase0']['model']
    seed_everything(cfg['seed'])
    if threads:
        torch.set_num_threads(threads)
    op = model_operator(load_model(cfg['phase0'], root, device=device).eval(), device)
    scene = synthetic_scene(s3['aoi_px'], cfg['seed'])
    kwargs = dict(tile=s1['tile'], stride=s1['stride'], scale=p0['scale'], feather=s1['feather'])
    if reverse_tiles:                       # control only: same tiles, opposite accumulation order
        axis = tile_origins(scene.shape[-1], s1['tile'], s1['stride'])
        planned = [(r, c) for r in axis for c in axis]
        kwargs['origins'] = planned[::-1]
    return super_resolve_tiled(scene, None, op, **kwargs).array


def worker(cfg, root, threads: int) -> None:
    out = run_e1(cfg, root, 'cpu', threads)
    print(json.dumps({'sha256': array_sha256(out), 'shape': list(out.shape), 'threads': threads}))


def fresh_process_run(config_path: str, threads: int) -> dict:
    proc = subprocess.run([sys.executable, '-m', 'experiments.e3_hash_reruns', '--worker', '--threads', str(threads),
                           '--config', config_path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'worker failed (threads={threads}): {proc.stderr[-500:]}')
    return json.loads(proc.stdout.strip().splitlines()[-1])


def gpu_part(cfg, root) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise Blocked('GPU reruns need the RTX 4050 machine (an NVIDIA GPU with CUDA torch): '
                      f'torch.cuda.is_available() is False here (torch {torch.__version__}, macOS arm64). '
                      'CPU hashes do not stand in for GPU drift, and MPS was deliberately not substituted.',
                      evidence='real')
    runs = [run_e1(cfg, root, 'cuda', None) for _ in range(cfg['e3']['gpu_runs'])]
    cpu = run_e1(cfg, root, 'cpu', cfg['e3']['cpu_threads'][0])
    return {'status': 'MEASURED', 'device': torch.cuda.get_device_name(0), **compare_runs(runs),
            'cpu_vs_gpu_max_abs_delta': float(np.abs(runs[0].astype(np.float64) - cpu).max())}


def probe(cfg, root):
    """E3: seeded, deterministic CPU reruns in fresh processes; GPU reruns if CUDA exists."""
    s = cfg['e3']
    config_path = str(root / 'configs' / 'experiments.yaml')
    per_thread = {}
    for threads in s['cpu_threads']:
        runs = [fresh_process_run(config_path, threads) for _ in range(s['cpu_runs'])]
        hashes = [r['sha256'] for r in runs]
        per_thread[str(threads)] = {'sha256': hashes, 'identical': len(set(hashes)) == 1}
    cpu_ok = all(v['identical'] for v in per_thread.values())
    across = len({v['sha256'][0] for v in per_thread.values()}) == 1
    order = compare_runs([run_e1(cfg, root, 'cpu', s['cpu_threads'][0]),
                          run_e1(cfg, root, 'cpu', s['cpu_threads'][0], reverse_tiles=True)])
    try:
        gpu = gpu_part(cfg, root)
    except Blocked as exc:
        gpu = {'status': 'BLOCKED', 'reason': str(exc)}
    return {'status': 'PASS' if cpu_ok else 'FAIL', 'evidence': 'synthetic',
            'evidence_parts': {'cpu_hash_reruns': 'PASS' if cpu_ok else 'FAIL', 'gpu_reruns': gpu['status']},
            'criteria': {'rule': 'CPU SHA-256 identical across reruns (fresh process, fixed threads); GPU drift is a '
                                 'finding, not a failure', 'aoi_px': s['aoi_px'], 'cpu_runs_per_thread_setting': s['cpu_runs']},
            'measurements': {'cpu_by_thread_count': per_thread,
                             'cpu_hash_identical_across_thread_counts': across,
                             'sensitivity_control_reversed_tile_order': {
                                 'hash_changes': not order['identical'], 'max_abs_delta': order['max_abs_delta'],
                                 'note': 'same tiles, opposite accumulation order; shows the hash can change, '
                                         'so a fixed tile order is a real requirement'},
                             'gpu': gpu,
                             'determinism_settings': ['torch.use_deterministic_algorithms(True)',
                                                      'cudnn.deterministic=True, benchmark=False',
                                                      'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'row-major tile order']},
            'hash_scope': 'macOS-arm64-specific: valid only for this CPU architecture, OS and torch 2.8.0 macOS build',
            'limitations': ['Bitwise CPU reproducibility is shown on this machine, OS and torch build only; it does not '
                            'transfer to other CPUs, BLAS builds or thread counts (see the cross-thread finding).',
                            'GPU determinism is untested (no CUDA here); it is BLOCKED until run on the RTX 4050 machine, not passed.',
                            'Synthetic scene; hash equality says nothing about accuracy.']}


if __name__ == '__main__':
    if '--worker' in sys.argv:
        from experiments.common import load_experiment_config
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker', action='store_true')
        parser.add_argument('--threads', type=int, required=True)
        parser.add_argument('--config', default='configs/experiments.yaml')
        args = parser.parse_args()
        cfg_, root_, _ = load_experiment_config(args.config)
        worker(cfg_, root_, args.threads)
    else:
        run_cli('e3', probe)
