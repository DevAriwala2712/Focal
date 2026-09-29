"""Measure the fixed-mask workaround under a 4 GiB allocator cap."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from risk.common import load_config, write_json
from trustsr.sr import load_finetuned_model, super_resolve


def main():
    cfg, root, _ = load_config('configs/pipeline.yaml')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1, cfg['model']['budget_gib']*2**30/device.total_memory))
    model = load_finetuned_model(cfg, root).eval()
    rng = np.random.default_rng(2024)
    measurements = []
    for size in [64, 128, 512]:
        tile = rng.uniform(.05, .5, (4, size, size)).astype('float32')
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = super_resolve(tile, model, tile_size=min(128, size), min_tile=64)
        torch.cuda.synchronize()
        measurements.append({'input_size': size, 'output_size': output.shape[-1],
                             'seconds': time.perf_counter()-start,
                             'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                             'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
    result = {'status': 'PASS', 'device': device.name, 'device_memory_mib': device.total_memory/2**20,
              'allocator_cap_gib': cfg['model']['budget_gib'], 'physical_4gb_device_verified': False,
              'input': 'seeded synthetic shape/memory probe', 'method': 'SEN2SR pretrained CNN tiled with parent-mean consistency projection',
              'measurements': measurements}
    write_json(root/'results'/'gpu_tiles.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
