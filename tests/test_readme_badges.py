import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_badges_match_result_files():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "update_badges.py"), "--check"], cwd=ROOT)
    assert r.returncode == 0, "README badges are stale: run python scripts/update_badges.py"
