import json
import os
import subprocess
import sys
from pathlib import Path


def test_missing_real_pairs_produces_blocked_report_and_nonzero_exit(tmp_path):
    import yaml
    cfg = yaml.safe_load(Path('configs/phase0.yaml').read_text(encoding='utf-8'))
    directory = tmp_path / 'configs'
    directory.mkdir()
    path = directory / 'risk.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in ['COLAB_GPU', 'COLAB_RELEASE_TAG']}
    result = subprocess.run([sys.executable, '-m', 'risk.r5_finetune', '--config', str(path)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    report = json.loads((tmp_path / 'risk/results/r5.json').read_text())
    assert report['status'] == 'BLOCKED'
    assert report['training_steps_run'] == 0
    assert 'manifest missing' in report['reason']
