"""Regenerate the README status badges from committed result files.

Each badge is computed from evidence and links to it:
  Phase 0      risk/results/r1..r5.json `status`
  Experiments  experiments/results/f1,f2,f3,f5,f6,f7.json `status`
  Phase 1      presence of trustsr/{fetch,sr,trust,change,evaluate}.py
  Demo         presence of demo/app.py (the Streamlit app)
  Event        risk/results/r2.json thresholds.event_date
Usage: python scripts/update_badges.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- badges:start -->", "<!-- badges:end -->"
RISKS = ["r1", "r2", "r3", "r4", "r5"]
EXPERIMENTS = ["f1", "f2", "f3", "f5", "f6", "f7"]
PHASE1 = ["fetch", "sr", "trust", "change", "evaluate"]


def status(rel: str) -> str:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))["status"]


def colour(passed: int, total: int) -> str:
    return "brightgreen" if passed == total else ("orange" if passed else "red")


def badge(label: str, message: str, col: str, link: str) -> str:
    esc = lambda s: quote(s.replace("-", "--").replace("_", "__"), safe="")
    src = f"https://img.shields.io/badge/{esc(label)}-{esc(message)}-{col}"
    return f'  <a href="{link}"><img alt="{label}: {message}" src="{src}"></a>'


def build() -> str:
    r = [status(f"risk/results/{k}.json") for k in RISKS]
    e = [status(f"experiments/results/{k}.json") for k in EXPERIMENTS]
    rp, ep = r.count("PASS"), e.count("PASS")
    p1 = sum((ROOT / "trustsr" / f"{m}.py").exists() for m in PHASE1)
    demo = (ROOT / "demo" / "app.py").exists()
    ev = date.fromisoformat(json.loads((ROOT / "risk/results/r2.json").read_text())["thresholds"]["event_date"])
    lines = [
        badge("Phase 0", f"{rp} of {len(r)} pass", colour(rp, len(r)), "RISK_REPORT.md"),
        badge("experiments F1–F7", f"{ep} of {len(e)} pass", colour(ep, len(e)), "RESULTS_V2.md"),
        badge("Phase 1 modules", f"{p1} of {len(PHASE1)} built" if p1 else "not started",
              colour(p1, len(PHASE1)) if p1 else "lightgrey", "trustsr/"),
        badge("demo", "built" if demo else "not built", "brightgreen" if demo else "lightgrey", "demo/"),
        badge("python", "3.10+", "blue", "requirements-phase0.txt"),
        badge("showcase", f"Wayanad {ev.day} {ev:%b %Y}", "red", "risk/results/r2.json"),
    ]
    return "\n".join([START, '<p align="center">', *lines, "</p>", END])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if README badges are stale")
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        sys.exit(f"README has no {START} … {END} block")
    new = pattern.sub(lambda _: build(), text)
    if ap.parse_args().check:
        return int(new != text)
    README.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
