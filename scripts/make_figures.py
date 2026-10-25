"""Render README figures from committed result JSON files.

Every number plotted is read from risk/results/ or experiments/results/; nothing is typed in by hand.
Usage: python scripts/make_figures.py [--out docs/assets]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b9b8b2"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.axisbelow": True,
})


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def caption(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK_2, ha="left", va="bottom")


def fig_false_alarms(out: Path) -> None:
    f1, f2 = load("experiments/results/f1.json"), load("experiments/results/f2.json")
    g = f1["gates"]
    rows = [
        ("Ungated SR", g["ungated_S_v1_sigma"]["test_split"], ORANGE),
        ("Gate v1 (SR + 10 m parent)", g["gate_v1"]["test_split"], BLUE),
        ("10 m rule only", g["rule_10m"]["test_split"], AQUA),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"wspace": 0.55})
    for i, (name, t, c) in enumerate(rows):
        p = t["pooled_pixel_far"]
        a.barh(i, p["estimate"], color=c, height=0.55)
        a.errorbar(p["estimate"], i, xerr=[[p["estimate"] - max(p["lo"], 1e-6)], [p["hi"] - p["estimate"]]],
                   fmt="none", ecolor=INK_2, capsize=3, lw=1)
        a.text(p["hi"] * 1.4, i, f"{p['estimate']:.2%}", va="center", fontsize=9.5, color=INK)
    a.set_xscale("log")
    a.set_xlim(1e-5, 3)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.invert_yaxis()
    a.set_xlabel("pixel false-alarm rate on no-event data (log scale)")
    a.set_title("F1 placebo test: the gate removes ~99.9% of false alarms")
    a.grid(axis="y", visible=False)

    w = f2["scoring_in_f1_harness"]["normalised_f3"]["pooled_window_far_test"]
    wrows = [("Ungated SR", rows[0][1]["pooled_window_far"], ORANGE),
             ("Gate v1", rows[1][1]["pooled_window_far"], BLUE),
             ("Gate v2", w, RED)]
    for i, (name, p, c) in enumerate(wrows):
        b.bar(i, p["estimate"], color=c, width=0.55)
        b.errorbar(i, p["estimate"], yerr=[[p["estimate"] - p["lo"]], [p["hi"] - p["estimate"]]],
                   fmt="none", ecolor=INK_2, capsize=3, lw=1)
        b.text(i, max(p["hi"], 0.05) + 0.025, f"{p['estimate']:.1%}", ha="center", fontsize=9.5)
    b.axhline(0.05, color=INK, lw=1.2, ls="--")
    b.text(1.5, 0.22, "dashed line = 5% ceiling", ha="center", fontsize=9, color=INK)
    b.set_xticks(range(3), [r[0] for r in wrows])
    b.set_ylim(0, 1)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    b.set_ylabel("window false-alarm rate")
    b.set_title("Window FAR vs the pre-registered ceiling")
    b.grid(axis="x", visible=False)
    caption(fig, "Source: experiments/results/f1.json, f2.json · test split · 95% block-bootstrap CIs · "
                 "Gate v2 = 6.5%, above the 5% rule (FAIL)")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(out / "fig_false_alarms.png", dpi=160)
    plt.close(fig)


def fig_sr_vs_bicubic(out: Path) -> None:
    f6 = load("experiments/results/f6.json")
    per, hal = f6["per_dataset"], f6["opensr_hallucination_column"]
    names = list(per)
    pooled = f6["pooled_all_datasets"]["recomputed"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"wspace": 0.35})
    labels = [f"{n.replace('_', ' ')}  (n={per[n]['n_images']})" for n in names] + [f"pooled  (n={pooled['n']})"]
    vals = [per[n]["delta_vs_bicubic_psnr_db"] for n in names] + [pooled["mean"]]
    los = [per[n]["ci95_lo"] for n in names] + [pooled["lo"]]
    his = [per[n]["ci95_hi"] for n in names] + [pooled["hi"]]
    for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
        c = AQUA if lo > 0 else (RED if hi < 0 else MUTED)
        a.barh(i, v, color=c, height=0.55)
        a.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK_2, capsize=3, lw=1)
        a.text(hi + 0.03 if v >= 0 else lo - 0.03, i, f"{v:+.2f} dB", va="center",
               ha="left" if v >= 0 else "right", fontsize=9)
    a.axvline(0, color=INK, lw=1)
    a.set_yticks(range(len(labels)), labels)
    a.invert_yaxis()
    a.set_xlim(-0.95, 0.55)
    a.set_xlabel("PSNR gain of SEN2SR-lite over bicubic (dB)")
    a.set_title("SR vs bicubic: no pooled gain")
    a.grid(axis="y", visible=False)

    x = range(len(names))
    sr = [hal[n]["sr_ha_metric"] for n in names]
    bi = [hal[n]["bicubic_ha_metric"] for n in names]
    b.bar([i - 0.18 for i in x], bi, width=0.34, color=BLUE, label="bicubic")
    b.bar([i + 0.18 for i in x], sr, width=0.34, color=ORANGE, label="SEN2SR-lite")
    b.set_xticks(list(x), [n.replace("_", "\n") for n in names])
    b.set_ylabel("opensr-test hallucination score (lower is better)")
    b.set_title("SR invents more detail on 5 of 5 datasets")
    b.legend(frameon=False, loc="upper left")
    b.grid(axis="x", visible=False)
    caption(fig, "Source: experiments/results/f6.json · green = CI above 0, red = CI below 0, grey = spans 0")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(out / "fig_sr_vs_bicubic.png", dpi=160)
    plt.close(fig)


def fig_cloud_timeline(out: Path) -> None:
    r2 = load("risk/results/r2.json")
    rows = load("risk/results/r2_acquisitions.json")["rows"]
    d = [date.fromisoformat(r["date"]) for r in rows]
    cs = [r["cloud_shadow_pct_aoi"] for r in rows]
    t = r2["thresholds"]
    thr = float(t["max_cloud_shadow_pct"])
    ok = [c <= thr and r["coverage_pct"] >= t["min_coverage_pct"] for c, r in zip(cs, rows)]
    fig, a = plt.subplots(figsize=(11, 3.6))
    a.vlines(d, 0, cs, color=GRID, lw=2)
    a.scatter([x for x, k in zip(d, ok) if not k], [c for c, k in zip(cs, ok) if not k],
              s=22, color=MUTED, label="cloudy / partial", zorder=3)
    a.scatter([x for x, k in zip(d, ok) if k], [c for c, k in zip(cs, ok) if k],
              s=42, color=AQUA, edgecolor=SURFACE, lw=1.5, label="clear (usable)", zorder=4)
    a.axhline(thr, color=INK_2, lw=1, ls="--")
    a.text(d[0], thr + 3, f"clear ≤ {thr:g}% cloud+shadow", fontsize=8.5, color=INK_2)
    ev = date.fromisoformat(t["event_date"])
    a.axvline(ev, color=RED, lw=1.6)
    a.text(ev, 104, f" landslide {ev:%d %b}", color=RED, fontsize=9.5, va="bottom")
    post = date.fromisoformat(r2["first_post_date"])
    a.annotate(f"first clear post-event\n{post:%d %b} (+{(post - ev).days} days)", xy=(post, 2), xytext=(date(2024, 9, 20), 55),
               fontsize=9, arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    a.axvspan(date(2024, 6, 1), date(2024, 9, 30), color=BLUE, alpha=0.06, lw=0)
    a.text(date(2024, 6, 5), 88, "SW monsoon (approx.)", color=BLUE, fontsize=9)
    a.set_ylim(0, 112)
    a.set_ylabel("AOI cloud + shadow (%)")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a.set_title("R2 · Sentinel-2 L2A over Wayanad, 2024: every acquisition, SCL-based cloud cover")
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    a.grid(axis="x", visible=False)
    caption(fig, f"Source: risk/results/r2_acquisitions.json · {len(rows)} acquisitions, 10×10 km AOI at 11.782°N 76.233°E")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.17)
    fig.savefig(out / "fig_cloud_timeline.png", dpi=160)
    plt.close(fig)


def fig_noise_coverage(out: Path) -> None:
    k = load("experiments/results/f3.json")["measurements"]["wayanad"]["keep_rule"]
    nominal = 0.9545
    rows = [("without normalisation", k["coverage_without_normalisation_same_sigma_method"],
             k["ci95_without_normalisation"], MUTED),
            ("noise model v2 (normalised)", k["coverage"], k["ci95"], BLUE)]
    fig, a = plt.subplots(figsize=(8, 2.6))
    a.axvspan(nominal - 0.02, nominal + 0.02, color=AQUA, alpha=0.15, lw=0)
    a.axvline(nominal, color=AQUA, lw=1.5)
    a.text(nominal, 1.62, "target 95.45% ± 2 pp", ha="center", fontsize=9, color=INK_2)
    for i, (name, v, ci, c) in enumerate(rows):
        a.plot(ci, [i, i], color=INK_2, lw=1.2)
        a.scatter([v], [i], s=90, color=c, edgecolor=SURFACE, lw=2, zorder=3)
        a.text(v, i + 0.28, f"{v:.1%}", ha="center", fontsize=9.5)
    a.set_yticks([0, 1], [r[0] for r in rows])
    a.set_ylim(-0.6, 1.9)
    a.set_xlim(0.82, 1.0)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_xlabel("share of stable pixels with |d| ≤ 2σ")
    a.set_title("F3 · Is the uncertainty σ honest? Yes, after normalisation")
    a.grid(axis="y", visible=False)
    caption(fig, "Source: experiments/results/f3.json · leave-one-date-out, 7.07 M stable 2.5 m pixels")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.85, bottom=0.3)
    fig.savefig(out / "fig_noise_coverage.png", dpi=160)
    plt.close(fig)


def fig_wayanad_classes(out: Path) -> None:
    cr = load("experiments/results/f7.json")["class_report"]
    order = [("CORE", RED), ("ALLOCATED", ORANGE), ("UNSUPPORTED", "#6250d6"), ("NO_DATA", MUTED)]
    fig, a = plt.subplots(figsize=(8, 2.8))
    for i, (name, c) in enumerate(order):
        v = cr[name]["km2"]
        a.barh(i, v, color=c, height=0.55)
        a.text(v + 0.02, i, f"{v:.2f} km²", va="center", fontsize=9.5)
    a.set_yticks(range(len(order)), [n.replace("_", " ").title() for n, _ in order])
    a.invert_yaxis()
    a.set_xlim(0, 1.8)
    a.set_xlabel("area (km²) in the 32.8 km² Wayanad crop")
    a.set_title("F7 · Wayanad change classes (gate v2, NOT validated)")
    a.grid(axis="y", visible=False)
    caption(fig, f"Source: experiments/results/f7.json · No-change {cr['NO_CHANGE']['km2']:.1f} km² not shown · "
                 "gate failed its FAR rule")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.27)
    fig.savefig(out / "fig_wayanad_classes.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_false_alarms, fig_sr_vs_bicubic, fig_cloud_timeline, fig_noise_coverage, fig_wayanad_classes):
        f(out)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
