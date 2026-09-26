"""Load the inspected, revision- and hash-pinned upstream SEN2SR-lite API."""
from __future__ import annotations

import importlib.util

from risk.common import digest, download


def enable_training_path(sr_model):
    # The inspected upstream trainable loader constructs Conv3XC(train_mode=False).
    # That branch calls update_params() using detached .data and freezes eval_conv.
    # .train() alone does not change this separate upstream flag.
    from sen2sr.models.opensr_baseline.cnn import Conv3XC
    for module in sr_model.modules():
        if isinstance(module, Conv3XC):
            module.train_mode = True


def verify_artifacts(directory, expected):
    if not expected:
        raise ValueError('Model artifact hashes must be configured before executing loader code')
    for name, sha in expected.items():
        path = directory / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f'Model artifact hash mismatch: {name}')


def load_model(cfg, root, *, trainable=False, device='cuda'):
    settings = cfg['model']
    directory = root / cfg['paths']['model']
    expected = settings['sha256']
    for name, sha in expected.items():
        download(settings['base_url'] + '/' + name, directory / name, cfg['network'], sha256=sha)
    verify_artifacts(directory, expected)
    spec = importlib.util.spec_from_file_location('trustsr_pinned_sen2sr_loader', directory / 'load.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model = (loader.trainable_model if trainable else loader.compiled_model)(directory, device=device)
    if trainable:
        enable_training_path(model.sr_model)
    if trainable and not any(p.requires_grad for p in model.sr_model.parameters()):
        raise RuntimeError('Upstream training loader returned frozen SR weights')
    return model
