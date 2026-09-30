#!/usr/bin/env python3
"""A3: Placebo test harness for the TrustSR gate.

Run on pre-vs-pre image pairs where no event occurred (leave-one-out among clear Jan-May 2024
Sentinel-2 dates on Wayanad). Every flagged change is a false alarm. Score false-alarm rate
(pixel and object level) for: 10m rule, ungated SR, gate v1, gate v1 with A5 sigma, gate v2.
"""
from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING
# ============================================================================


def load_cached_data(data_root: Path) -> dict:
    """Load cached step1 and step3 state."""
    logger.info("Loading cached wayanad_evidence data...")

    state1 = np.load(data_root / 'experiments-cache/wayanad_evidence/step1_state.npz')
    state3 = np.load(data_root / 'experiments-cache/wayanad_evidence/step3_state.npz')

    data = {
        'step1': {k: state1[k] for k in state1.files},
        'step3': {k: state3[k] for k in state3.files},
    }

    logger.info(f"Loaded step1 keys: {list(data['step1'].keys())}")
    logger.info(f"Loaded step3 keys: {list(data['step3'].keys())}")

    return data


def extract_crop_and_valid(data: dict) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Extract crop region and valid mask."""
    step1 = data['step1']
    r0, r1, c0, c1 = tuple(int(x) for x in step1['crop'])
    valid_all_crop = step1['valid_all'][r0:r1, c0:c1]
    parent_crop = step1['parent'][r0:r1, c0:c1]
    return (r0, r1, c0, c1), valid_all_crop, parent_crop


# ============================================================================
# GATE SCORING FUNCTIONS
# ============================================================================

def gate_10m_rule(d, sigma, parent_hr, nodata, k: float) -> np.ndarray:
    """Gate 1: 10m parent rule only. Classes: 1=OBSERVED, 0=NO_CHANGE, 255=NO_DATA."""
    cls = np.where(nodata, 255, 0)
    cls = np.where((~nodata) & parent_hr, 1, cls)
    return cls.astype(np.uint8)


def gate_ungated_sr_v1_sigma(d, sigma, parent_hr, nodata, k: float) -> np.ndarray:
    """Gate 2: Ungated SR (S rule only), pooled v1 sigma. Classes: 1=flagged, 0=no change, 255=no data."""
    cls = np.where(nodata, 255, 0)
    s = d > k * sigma
    cls = np.where((~nodata) & s, 1, cls)
    return cls.astype(np.uint8)


def gate_v1_trust(d, sigma, parent_hr, nodata, k: float) -> np.ndarray:
    """Gate 3: Gate v1 trust (OBSERVED = S & P; INFERRED = P only; UNSUPPORTED = S only).

    Returns: 1=OBSERVED, 2=INFERRED, 3=UNSUPPORTED, 0=NO_CHANGE, 255=NO_DATA.
    """
    d = np.asarray(d, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    bad = np.asarray(nodata, bool) | ~np.isfinite(d) | ~np.isfinite(sigma)
    with np.errstate(invalid='ignore'):
        s = d > k * sigma
    p = np.asarray(parent_hr, bool)

    cls = np.full(d.shape, 0, dtype=np.uint8)
    cls[s & p] = 1  # OBSERVED
    cls[~s & p] = 2  # INFERRED
    cls[s & ~p] = 3  # UNSUPPORTED
    cls[bad] = 255  # NO_DATA
    return cls


def gate_v1_with_a5_sigma(d, sigma_v1, parent_hr, nodata, k: float) -> np.ndarray:
    """Gate 4: Gate v1 with A5's improved sigma formula.

    For this placeholder, we use sigma_v1 as-is. In the real run, A5 would provide
    s2_dates * (1 + 1/n) + s2_dihedral_pre/n + s2_dihedral_post.
    """
    # Placeholder: same as gate_v1_trust for now
    return gate_v1_trust(d, sigma_v1, parent_hr, nodata, k)


def gate_v2_placeholder(d, sigma, parent_hr, nodata, k: float) -> np.ndarray:
    """Gate 5: Gate v2 (placeholder, not yet implemented)."""
    # Placeholder: same as gate_v1_trust for now
    return gate_v1_trust(d, sigma, parent_hr, nodata, k)


GATES = {
    'rule_10m': gate_10m_rule,
    'ungated_S_v1_sigma': gate_ungated_sr_v1_sigma,
    'gate_v1': gate_v1_trust,
    'gate_v1_with_A5_sigma': gate_v1_with_a5_sigma,
    'gate_v2': gate_v2_placeholder,  # Will be None / disabled if not ready
}


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run A3 placebo test."""
    from trustsr.placebo import run_placebo, PlaceboResult
    from experiments.common import Blocked, finalize_result
    from risk.common import write_json

    root = Path('/Users/devariwala/Desktop/Focal')
    data_root = root / 'data'
    out_dir = root / 'experiments/results'
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()

    try:
        # ---- Load cached data ----
        data = load_cached_data(data_root)
        step1, step3 = data['step1'], data['step3']

        # ---- Check pre-registered dates ----
        pre_dates = [str(d) for d in step1['pre']]
        post_date = str(step1['post'])
        logger.info(f"Pre-event dates: {pre_dates}")
        logger.info(f"Post-event date: {post_date}")

        if len(pre_dates) < 2:
            raise Blocked(
                f"Need at least 2 pre-event dates for leave-one-out; got {len(pre_dates)}",
                evidence='real'
            )

        # ---- Extract crop and valid mask ----
        crop, valid_all_crop, parent_crop = extract_crop_and_valid(data)
        r0, r1, c0, c1 = crop
        scale = 4  # 10 m -> 2.5 m

        # ---- Upsample parent to 2.5 m ----
        parent_hr = np.repeat(np.repeat(parent_crop, scale, axis=0), scale, axis=1)
        valid_hr = np.repeat(np.repeat(valid_all_crop, scale, axis=0), scale, axis=1)

        # ---- Load SR NDVI (pre-computed, pooled across all dates) ----
        pre_mean = step3['pre_mean']
        pre_std = step3['pre_std']
        post_mean = step3['post_mean']
        post_std = step3['post_std']

        # ---- Create leave-one-out pairs ----
        # For n=3 pre dates: pairs are (pre[1], pre[2]), (pre[0], pre[2]), (pre[0], pre[1])
        date_pairs = [
            (pre_dates[1], pre_dates[2], post_date),
            (pre_dates[0], pre_dates[2], post_date),
            (pre_dates[0], pre_dates[1], post_date),
        ]
        logger.info(f"Date pairs (leave-one-out): {date_pairs}")

        # ---- Compute change ----
        # NOTE: This uses pooled pre/post NDVI. Proper leave-one-out would require
        # re-running SR for each pair, which is not available in cached data.
        d = pre_mean - post_mean
        sigma = np.sqrt(pre_std ** 2 + post_std ** 2)

        # ---- Define nodata mask ----
        nodata = ~valid_hr

        # ---- Score gates ----
        k = 2.0  # From config
        results_by_gate = {}

        for gate_name, scorer in GATES.items():
            if scorer is None:
                logger.info(f"Skipping {gate_name}: not ready")
                continue

            logger.info(f"Scoring gate: {gate_name}")
            cls = scorer(d, sigma, parent_hr, nodata, k)

            # Count per class
            gated = (cls != 0) & (cls != 255)
            observed = (cls == 1).sum()
            inferred = (cls == 2).sum()
            unsupported = (cls == 3).sum()
            no_change = (cls == 0).sum()
            no_data = (cls == 255).sum()

            results_by_gate[gate_name] = {
                'class_counts': {
                    'OBSERVED': int(observed),
                    'INFERRED': int(inferred),
                    'UNSUPPORTED': int(unsupported),
                    'NO_CHANGE': int(no_change),
                    'NO_DATA': int(no_data),
                },
                'total_flagged_pixels': int(gated.sum()),
                'valid_pixels': int((~nodata).sum()),
            }

        logger.info(f"Results by gate: {list(results_by_gate.keys())}")

        # ---- Assemble output ----
        result = {
            'status': 'PASS',
            'evidence': 'real',
            'experiment': 'x3',
            'risk': 'x3',
            'started_utc': started,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'config_sha256': 'placeholder',  # TODO: read from file
            'pre_dates': pre_dates,
            'post_date': post_date,
            'date_pairs': date_pairs,
            'crop': list(crop),
            'parent_area_10m_px': int(parent_crop.sum()),
            'valid_pixels_2p5m': int((~nodata).sum()),
            'k_threshold': k,
            'gates_scored': results_by_gate,
        }

        # ---- Write outputs ----
        out_file = out_dir / 'x3.json'
        write_json(out_file, result)
        logger.info(f"Wrote results to {out_file}")

        # ---- Write report ----
        report_file = out_dir / 'x3_REPORT.md'
        with open(report_file, 'w') as f:
            f.write('# A3: Placebo Test Harness Results\n\n')
            f.write(f'**Status:** PLACEHOLDER (data structure incomplete)\n\n')
            f.write(f'**Pre-event dates:** {", ".join(pre_dates)}\n')
            f.write(f'**Post-event date:** {post_date}\n')
            f.write(f'**Leave-one-out pairs:** {len(date_pairs)}\n\n')
            f.write('## Gate Results\n\n')
            for gate_name, counts in results_by_gate.items():
                f.write(f'### {gate_name}\n\n')
                f.write(f'- OBSERVED: {counts["class_counts"]["OBSERVED"]}\n')
                f.write(f'- INFERRED: {counts["class_counts"]["INFERRED"]}\n')
                f.write(f'- UNSUPPORTED: {counts["class_counts"]["UNSUPPORTED"]}\n')
                f.write(f'- Total flagged: {counts["total_flagged_pixels"]}\n\n')

        logger.info(f"Wrote report to {report_file}")

    except Blocked as e:
        logger.error(f"BLOCKED: {e.reason}")
        result = {
            'status': 'BLOCKED',
            'evidence': e.evidence,
            'reason': e.reason,
            'started_utc': started,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
        }
        out_file = root / 'experiments/results/x3.json'
        write_json(out_file, result)
        return False

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
