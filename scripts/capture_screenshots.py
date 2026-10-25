"""Capture README screenshots of the running web demo.

Start the demo first (python scripts/build_web_data.py && python demo/server.py), then:
    pip install playwright==1.47.0 && python -m playwright install chromium
    python scripts/capture_screenshots.py --base-url http://127.0.0.1:8765

Writes docs/media/screenshots/<page>.png. Pages that fail to load are reported, not faked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["how-it-works", "demo", "map-workspace", "evidence", "experiments", "data-dates", "objects", "exports"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "screenshots"))
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name in PAGES:
            page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=2)
            resp = page.goto(f"{args.base_url}/{name}.html", wait_until="networkidle")
            if resp is None or not resp.ok:
                failed.append(name)
                page.close()
                continue
            page.wait_for_timeout(1500)
            page.screenshot(path=str(out / f"{name}.png"))
            print("wrote", name)
            page.close()
        browser.close()
    if failed:
        print("failed to load:", ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
