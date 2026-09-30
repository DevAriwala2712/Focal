"""Pretrained SEN2SR-lite inference: native 128 px inputs, 8 dihedral variants per tile, ADR-001 OOM policy."""
from __future__ import annotations

import resource
import sys

import numpy as np

from experiments.wayanad_evidence.stats import dihedral, inverse_dihedral


def infer_with_oom_policy(fn, batch, tile_name, oom_type, cleanup, memory):
    """Run `fn(batch)`. On OOM: empty the cache and retry ONCE at batch 1; a second OOM fails naming the tile (no 64 px path)."""
    try:
        return fn(batch)
    except oom_type:
        cleanup()
        try:
            return np.concatenate([fn(batch[i:i + 1]) for i in range(len(batch))])
        except oom_type:
            raise RuntimeError(f'CUDA OOM at {tile_name} even at batch 1 after emptying the cache; '
                               f'memory counters: {memory()}') from None


def sr_variants(model_fn, tile, tile_name, oom_type, cleanup, memory):
    """(4,128,128) tile -> (8,4,512,512): the SR of each dihedral variant, inverted back to the original orientation."""
    batch = np.stack([dihedral(tile, k) for k in range(8)])
    out = infer_with_oom_policy(model_fn, batch, tile_name, oom_type, cleanup, memory)
    return np.stack([inverse_dihedral(out[k], k) for k in range(8)])


def peak_rss_mib() -> float:
    """Peak resident set size of this process (CPU proxy only; NOT GPU memory). ru_maxrss is bytes on macOS, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == 'darwin' else raw / 1024


class Runner:
    """Callable numpy (B,4,128,128) reflectance -> (B,4,512,512); plus the OOM hooks the policy needs."""

    def __init__(self, cfg, root):
        import torch
        from risk.model import load_model
        from experiments.common import Blocked
        device = cfg['model']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise Blocked('device cuda requested but torch.cuda.is_available() is False', evidence='real')
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        if cfg['model']['deterministic']:
            torch.use_deterministic_algorithms(True)
        self.torch, self.device, self.tile = torch, device, cfg['tiling']['tile']
        self.model = load_model(cfg['phase0'], root, trainable=False, device=device)
        self.oom_type = torch.OutOfMemoryError
        self.hard_constraint_mask = self.model.hard_constraint.low_pass_mask.detach().cpu().numpy()

    def __call__(self, batch):
        if batch.shape[-2:] != (self.tile, self.tile) or batch.shape[1] != 4:
            raise ValueError(f'model accepts only (B,4,{self.tile},{self.tile}); got {batch.shape}')
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
            return self.model(x).float().cpu().numpy()

    def cleanup(self):
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()

    def memory(self):
        if self.device == 'cuda':
            t = self.torch.cuda
            return {'allocated_mib': t.memory_allocated() / 2**20, 'reserved_mib': t.memory_reserved() / 2**20,
                    'peak_allocated_mib': t.max_memory_allocated() / 2**20, 'peak_reserved_mib': t.max_memory_reserved() / 2**20}
        return {'peak_rss_mib': peak_rss_mib(), 'gpu_memory': 'not measured: no CUDA device on this machine'}
