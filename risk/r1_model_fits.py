"""Measure real pretrained inference; separate shape failure from CUDA OOM."""
from __future__ import annotations

import gc
import statistics
import time

from risk.common import retry_tiles, run_cli
from risk.model import load_model


def benchmark(model, size, settings, torch):
    def once():
        # Synthetic tensor is ONLY a memory/latency fixture, never a fidelity result.
        x = torch.rand((1, 4, size, size), device='cuda', dtype=torch.float32)
        with torch.inference_mode():
            y = model(x)
        torch.cuda.synchronize()
        if list(y.shape) != [1, 4, size * settings['scale'], size * settings['scale']]:
            raise ValueError(f'Unexpected output shape: {list(y.shape)}')
        if not torch.isfinite(y).all():
            raise ValueError('Non-finite model output')
        return list(y.shape)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(settings['warmups']):
        once()
    times = []
    for _ in range(settings['repeats']):
        torch.cuda.synchronize()
        start = time.perf_counter()
        shape = once()
        times.append(time.perf_counter() - start)
    free, total = torch.cuda.mem_get_info()
    return {'input_tile': size, 'output_shape': shape, 'status': 'PASS',
            'seconds_per_tile_median': statistics.median(times), 'seconds_samples': times,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'device_used_after_mib': (total - free) / 2**20}


def probe(cfg, root):
    """Run batch-one FP32 benchmarks under a 4 GiB PyTorch allocation budget."""
    import torch
    if not torch.cuda.is_available():
        return {'status': 'BLOCKED', 'reason': 'CUDA-enabled PyTorch and GPU required', 'torch': torch.__version__}
    settings = cfg['model']
    torch.manual_seed(settings['seed'])
    total = torch.cuda.get_device_properties(0).total_memory
    budget = settings['budget_gib'] * 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), 0)
    model = load_model(cfg, root).eval()
    results = []
    for size in settings['benchmark_sizes']:
        try:
            result = benchmark(model, size, settings, torch)
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            result = {'input_tile': size, 'status': 'FAIL', 'failure': 'CUDA_OOM', 'reason': str(exc)}
            try:
                fallback, used, failures = retry_tiles(lambda s: benchmark(model, s, settings, torch),
                    size // 2, settings['min_tile'], torch.cuda.OutOfMemoryError, torch.cuda.empty_cache)
                result['fallback'] = {'used_tile': used, 'oom_sizes': failures, 'measurement': fallback}
            except Exception as fallback_exc:
                result['fallback_error'] = str(fallback_exc)
        except Exception as exc:
            result = {'input_tile': size, 'status': 'FAIL', 'failure': 'MODEL_SHAPE_OR_RUNTIME', 'reason': str(exc)}
        results.append(result)
        print(f"R1 input {size}: {result['status']}", flush=True)
    base_ok = any(r['input_tile'] == settings['native_tile'] and r['status'] == 'PASS' for r in results)
    return {'status': 'PASS' if base_ok and all(r['status'] == 'PASS' for r in results) else 'FAIL',
            'native_tile_passed': base_ok, 'device': torch.cuda.get_device_name(0), 'total_vram_mib': total / 2**20,
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'precision': 'float32',
            'input_kind': 'seeded synthetic memory probe; not real imagery or fidelity evidence',
            'allocator_budget_gib': settings['budget_gib'], 'measurements': results,
            'physical_4gb_gpu_verified': False,
            'limitation': 'Allocator cap excludes CUDA context and other processes. This is not a physical 4 GB GPU test.',
            'model_revision': settings['revision'], 'artifact_sha256': settings['sha256']}


if __name__ == '__main__':
    run_cli('r1', probe)
