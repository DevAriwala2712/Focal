"""Generate real precomputed Wayanad tile COGs on the local GPU."""
from __future__ import annotations

import json

import torch

from risk.common import load_config
from trustsr.pipeline import process_stack
from trustsr.sr import load_finetuned_model, super_resolve


def main():
    cfg, root, _ = load_config('configs/pipeline.yaml')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; precomputed demo requires a GPU')
    fraction = cfg['model']['budget_gib']*2**30/torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, fraction))
    model = load_finetuned_model(cfg, root).eval()
    def operator(tile):
        return super_resolve(tile, model, device='cuda', tile_size=cfg['sr']['tile_size'],
                             overlap=cfg['sr']['overlap'], min_tile=cfg['sr']['min_tile'])
    result = process_stack(root / cfg['fetch']['stack'], root / cfg['demo']['output_dir'], operator,
                           event_date=cfg['imagery']['event_date'],
                           tile_pixels=cfg['demo']['tile_pixels'], k=cfg['change']['k'],
                           parent_drop_threshold=cfg['change']['parent_drop_threshold'],
                           invalid_scl=cfg['change']['invalid_scl'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
