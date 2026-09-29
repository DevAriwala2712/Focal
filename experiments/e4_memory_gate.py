"""E4: memory gate at native 128 px. Evidence for a HUMAN decision about the 64 px fallback; decides nothing."""
from __future__ import annotations

import contextlib

import numpy as np

from experiments.common import Blocked, run_cli, synthetic_scene
from experiments.e1_tile_scheduler import model_operator, super_resolve_tiled


class StageMemory:
    """Per-stage peak allocated/reserved bytes. `mem` exposes reset_peak(), max_allocated(), max_reserved()."""

    def __init__(self, mem):
        self.mem = mem
        self.stages: dict = {}

    def __call__(self, name: str):
        return self._stage(name)

    @contextlib.contextmanager
    def _stage(self, name: str):
        self.mem.reset_peak()
        try:
            yield
        finally:
            rec = self.stages.setdefault(name, {'calls': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0})
            rec['calls'] += 1
            rec['peak_allocated_bytes'] = max(rec['peak_allocated_bytes'], self.mem.max_allocated())
            rec['peak_reserved_bytes'] = max(rec['peak_reserved_bytes'], self.mem.max_reserved())

    def overall(self) -> dict:
        return {'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in self.stages.values()),
                'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in self.stages.values())}


def gate_verdict(peak_reserved_bytes: int, cap_gib: float) -> bool:
    """Keep rule: peak reserved strictly below the cap."""
    return peak_reserved_bytes < cap_gib * 2 ** 30


class _CudaMem:
    def __init__(self, torch):
        self.t = torch.cuda
    def reset_peak(self):
        self.t.reset_peak_memory_stats()
    def max_allocated(self):
        return self.t.max_memory_allocated()
    def max_reserved(self):
        return self.t.max_memory_reserved()


def probe(cfg, root):
    """E4: 1000x1000 px AOI, batch 1, allocator capped at 4 GiB, per-stage peak memory."""
    import torch
    if not torch.cuda.is_available():
        raise Blocked('CUDA GPU required: torch.cuda.is_available() is False on this machine '
                      f'(torch {torch.__version__}, no NVIDIA GPU/driver). Memory cannot be measured on CPU, and an '
                      'allocator cap is meaningless without CUDA. Needed: an NVIDIA GPU with a CUDA build of torch '
                      '(the project reference is an RTX 4050 Laptop 6 GB).', evidence='synthetic',
                      limitations=['The CUDA path of this experiment is implemented but was never executed on CUDA; only the '
                                   'stage-tracking logic is unit-tested, against a fake memory API.',
                                   'No memory number exists yet. Nothing here supports or opposes dropping the 64 px fallback.'])
    from risk.model import load_model
    s, s1, p0 = cfg['e4'], cfg['e1'], cfg['phase0']['model']
    torch.manual_seed(cfg['seed'])
    total = torch.cuda.get_device_properties(0).total_memory
    cap = s['cap_gib'] * 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
    rec = StageMemory(_CudaMem(torch))
    with rec('model_load'):
        model = load_model(cfg['phase0'], root, device='cuda').eval()
    scene = synthetic_scene(s['aoi_px'], cfg['seed'])
    out = super_resolve_tiled(scene, None, model_operator(model, 'cuda'), tile=s1['tile'], stride=s1['stride'],
                              scale=p0['scale'], feather=s1['feather'], batch_size=1, stage=rec)
    overall = rec.overall()
    ok = gate_verdict(overall['peak_reserved_bytes'], s['cap_gib'])
    mib = lambda b: b / 2 ** 20
    return {'status': 'PASS' if ok else 'FAIL', 'evidence': 'synthetic',
            'criteria': {'rule': 'peak reserved < cap', 'cap_gib': s['cap_gib'], 'aoi_px': s['aoi_px'], 'batch': 1},
            'measurements': {'device': torch.cuda.get_device_name(0), 'device_total_mib': mib(total),
                             'tiles': len(out.origins), 'torch': torch.__version__,
                             'stages_mib': {k: {'calls': v['calls'], 'peak_allocated_mib': mib(v['peak_allocated_bytes']),
                                                'peak_reserved_mib': mib(v['peak_reserved_bytes'])}
                                            for k, v in rec.stages.items()},
                             'overall_peak_allocated_mib': mib(overall['peak_allocated_bytes']),
                             'overall_peak_reserved_mib': mib(overall['peak_reserved_bytes']),
                             'device_used_after_mib': mib(total - torch.cuda.mem_get_info()[0])},
            'physical_4gb_gpu_verified': False,
            'limitations': ['A PyTorch allocator cap on a larger card is NOT a physical 4 GB GPU: it excludes the CUDA '
                            'context, cuDNN/cuBLAS workspaces and other processes.',
                            'Blend accumulators live in host RAM (numpy), so they are outside these GPU numbers.',
                            'Synthetic pixels: memory does not depend on pixel values, but this is not an accuracy result.',
                            'This is evidence for a human decision about the 64 px fallback; it does not make that decision.']}


if __name__ == '__main__':
    run_cli('e4', probe)
