import numpy as np
import pytest
from affine import Affine

from trustsr.sr import spectral_project, spectral_project_torch, tiled_super_resolve, sr_geometry, super_resolve


def test_parent_spectral_consistency_and_geometry():
    lr = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5) / 100
    guessed = np.full((4, 12, 20), .7, dtype=np.float32)
    result = spectral_project(lr, guessed)
    np.testing.assert_allclose(result.reshape(4, 3, 4, 5, 4).mean((2, 4)), lr, atol=1e-6)
    transform = Affine.translation(500000, 1200000) * Affine.scale(10, -10)
    out_transform, shape = sr_geometry(transform, lr.shape[-2:], 4)
    assert shape == (12, 20)
    assert out_transform == transform * Affine.scale(.25, .25)
    assert out_transform * (20, 12) == transform * (5, 3)


def test_overlapped_tiling_matches_single_pass_for_local_operator():
    source = np.arange(4 * 13 * 17, dtype=np.float32).reshape(4, 13, 17) / 10000
    def local(tile):
        return np.repeat(np.repeat(tile, 4, -2), 4, -1)
    np.testing.assert_allclose(tiled_super_resolve(source, local, tile_size=8, overlap=2), local(source), atol=1e-6)


def test_final_feathered_output_preserves_each_parent_pixel():
    import torch
    class BiasedNetwork(torch.nn.Module):
        def forward(self, x):
            ramp = .5 * x.mean() * (torch.arange(x.shape[-2] * 4, device=x.device) % 4)[None, None, :, None]
            return x.repeat_interleave(4, -2).repeat_interleave(4, -1) + x.mean() + ramp
    model = type('Model', (), {'sr_model': BiasedNetwork()})()
    source = np.linspace(.1, .8, 4 * 17 * 19, dtype=np.float32).reshape(4, 17, 19)
    result = super_resolve(source, model, device='cpu', tile_size=8, overlap=2)
    np.testing.assert_allclose(result.reshape(4, 17, 4, 19, 4).mean((2, 4)), source, atol=1e-6)


def test_cuda_oom_halves_tile_to_64_and_retries():
    import torch
    sizes = []
    class LimitedNetwork(torch.nn.Module):
        def forward(self, x):
            sizes.append(x.shape[-1])
            if x.shape[-1] > 64:
                raise torch.cuda.OutOfMemoryError('probe')
            return x.repeat_interleave(4, -2).repeat_interleave(4, -1)
    model = type('Model', (), {'sr_model': LimitedNetwork()})()
    source = np.full((4, 128, 128), .2, dtype=np.float32)
    result = super_resolve(source, model, device='cpu', tile_size=128, min_tile=64)
    assert sizes[0] == 128 and 64 in sizes
    np.testing.assert_allclose(result, .2, atol=1e-6)


def test_cuda_oom_at_minimum_tile_fails_clearly():
    import torch
    class FailingNetwork(torch.nn.Module):
        def forward(self, x):
            raise torch.cuda.OutOfMemoryError('probe')
    model = type('Model', (), {'sr_model': FailingNetwork()})()
    with pytest.raises(RuntimeError, match='minimum tile size 64'):
        super_resolve(np.ones((4, 64, 64), dtype=np.float32), model, device='cpu',
                      tile_size=64, min_tile=64)


def test_torch_parent_projection_is_exact_and_differentiable():
    import torch
    lr = torch.full((1, 4, 2, 3), .3)
    predicted = (torch.arange(1*4*8*12, dtype=torch.float32).reshape(1, 4, 8, 12) / 1000).requires_grad_()
    corrected = spectral_project_torch(lr, predicted)
    torch.testing.assert_close(corrected.reshape(1, 4, 2, 4, 3, 4).mean((3, 5)), lr, atol=1e-5, rtol=0)
    corrected.square().mean().backward()
    assert predicted.grad is not None and torch.count_nonzero(predicted.grad) > 0
