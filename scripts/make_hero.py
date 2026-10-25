"""Crop the four panels of the Wayanad evidence figure into a labelled README hero strip.

Input:  experiments/wayanad_evidence/outputs/figure.png (real Sentinel-2 / SR outputs, step 5 of the evidence run)
Output: docs/media/wayanad_hero.png
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "wayanad_evidence" / "outputs" / "figure.png"
OUT = ROOT / "docs" / "media" / "wayanad_hero.png"
LABELS = ["BEFORE · 10 m · Jan 2024", "AFTER · 10 m · 6 Dec 2024",
          "AFTER · 2.5 m SR (model reconstruction)", "TRUST GATE · observed change in red"]


def runs(idx):
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1:
            out.append((s, p))
            s = v
        p = v
    out.append((s, p))
    return [r for r in out if r[1] - r[0] > 200]


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    im = Image.open(SRC).convert("RGB")
    dark = np.asarray(im).astype(int).sum(2) < 120
    cols = runs(np.where(dark.mean(0) > 0.35)[0])
    rows = runs(np.where(dark.mean(1) > 0.25)[0])
    boxes = [(c0 + 2, r0 + 2, c1 - 1, r1 - 1) for (r0, r1) in rows for (c0, c1) in cols]
    panels = [im.crop(b) for b in boxes]
    w, h = panels[0].size
    gap, bar = 8, 34
    strip = Image.new("RGB", (4 * w + 3 * gap, h + bar), "#0d1117")
    d = ImageDraw.Draw(strip)
    f = font(17)
    for i, (p, label) in enumerate(zip(panels, LABELS)):
        x = i * (w + gap)
        strip.paste(p.resize((w, h)), (x, bar))
        d.text((x + 10, 8), label, fill="#f0efec", font=f)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    strip.save(OUT, optimize=True)
    print("wrote", OUT.relative_to(ROOT), strip.size)


if __name__ == "__main__":
    main()
