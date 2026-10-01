import json
import hashlib
from pathlib import Path

def file_sha256(path: Path) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def resolve_pointer(data, pointer):
    parts = [p for p in pointer.split('/') if p]
    cur = data
    for p in parts:
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur[p]
    return cur

def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / 'experiments' / 'results'
    
    claims = []
    
    # Claim F1: Ungated pixel FAR
    f1_path = results_dir / 'f1.json'
    f1_hash = file_sha256(f1_path)
    claims.append({
        "id": "F1.ungated.pixel_far",
        "plain_text": "ungated SR change detection flags",
        "technical_text": "Pixel false alarm rate for ungated SR change detection on a no-event pair",
        "unit": "pixel",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f1.json",
        "json_pointer": "/gates/ungated_S_v1_sigma/test_split/pooled_pixel_far/estimate",
        "source_sha256": f1_hash,
        "ci_95": {
            "pointer_lo": "/gates/ungated_S_v1_sigma/test_split/pooled_pixel_far/lo",
            "pointer_hi": "/gates/ungated_S_v1_sigma/test_split/pooled_pixel_far/hi"
        },
        "baseline": None,
        "denominator": "/gates/ungated_S_v1_sigma/test_split/pooled_pixel_far/denominator"
    })
    
    # Claim F1: Gated v1 pixel FAR
    claims.append({
        "id": "F1.gate_v1.pixel_far",
        "plain_text": "the gated product flags",
        "technical_text": "Pixel false alarm rate for gate v1 on a no-event pair",
        "unit": "pixel",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f1.json",
        "json_pointer": "/gates/gate_v1/test_split/pooled_pixel_far/estimate",
        "source_sha256": f1_hash,
        "ci_95": {
            "pointer_lo": "/gates/gate_v1/test_split/pooled_pixel_far/lo",
            "pointer_hi": "/gates/gate_v1/test_split/pooled_pixel_far/hi"
        },
        "baseline": None,
        "denominator": "/gates/gate_v1/test_split/pooled_pixel_far/denominator"
    })
    
    # Claim F2: Gate v2 window FAR
    f2_path = results_dir / 'f2.json'
    f2_hash = file_sha256(f2_path)
    claims.append({
        "id": "F2.window_far",
        "plain_text": "Gate v2 test-split window FAR",
        "technical_text": "Test-split window FAR for Gate v2 against a required ≤ 0.05",
        "unit": "window",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "FAIL",
        "source_file": "experiments/results/f2.json",
        "json_pointer": "/scoring_in_f1_harness/normalised_f3/pooled_window_far_test/estimate",
        "source_sha256": f2_hash,
        "ci_95": {
            "pointer_lo": "/scoring_in_f1_harness/normalised_f3/pooled_window_far_test/lo",
            "pointer_hi": "/scoring_in_f1_harness/normalised_f3/pooled_window_far_test/hi"
        },
        "baseline": 0.05,
        "denominator": "/scoring_in_f1_harness/normalised_f3/pooled_window_far_test/denominator"
    })
    
    # F3 coverage
    f3_path = results_dir / 'f3.json'
    f3_hash = file_sha256(f3_path)
    claims.append({
        "id": "F3.coverage.normalised",
        "plain_text": "Noise model v2 coverage",
        "technical_text": "Coverage of the noise model v2 keep rule",
        "unit": "percentage",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f3.json",
        "json_pointer": "/measurements/wayanad/keep_rule/coverage",
        "source_sha256": f3_hash,
        "ci_95": {
            "pointer_lo": "/measurements/wayanad/keep_rule/ci95/0",
            "pointer_hi": "/measurements/wayanad/keep_rule/ci95/1"
        },
        "baseline": 0.9545,
        "denominator": "/measurements/wayanad/keep_rule/n_px"
    })
    
    # F5 headline with NAIP
    f5_path = results_dir / 'f5.json'
    f5_hash = file_sha256(f5_path)
    claims.append({
        "id": "F5.iou.with_naip",
        "plain_text": "SR-informed ranking IoU with NAIP",
        "technical_text": "Headline +0.0081 IoU for SR-informed ranking with NAIP included",
        "unit": "IoU",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f5.json",
        "json_pointer": "/headline/oracle/all_datasets/mean",
        "source_sha256": f5_hash,
        "ci_95": {
            "pointer_lo": "/headline/oracle/all_datasets/lo",
            "pointer_hi": "/headline/oracle/all_datasets/hi"
        },
        "baseline": 0.0,
        "denominator": "/headline/oracle/all_datasets/n"
    })
    
    # F5 headline without NAIP
    claims.append({
        "id": "F5.iou.without_naip",
        "plain_text": "SR-informed ranking IoU excluding NAIP",
        "technical_text": "Headline IoU for SR-informed ranking with NAIP excluded",
        "unit": "IoU",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f5.json",
        "json_pointer": "/headline/oracle/excluding_naip/mean",
        "source_sha256": f5_hash,
        "ci_95": {
            "pointer_lo": "/headline/oracle/excluding_naip/lo",
            "pointer_hi": "/headline/oracle/excluding_naip/hi"
        },
        "baseline": 0.0,
        "denominator": "/headline/oracle/excluding_naip/n"
    })
    
    # F6 SR vs bicubic PSNR pooled
    f6_path = results_dir / 'f6.json'
    f6_hash = file_sha256(f6_path)
    claims.append({
        "id": "F6.psnr.pooled",
        "plain_text": "SR does not beat bicubic on PSNR pooled",
        "technical_text": "Pooled SR vs bicubic PSNR across all datasets",
        "unit": "dB",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f6.json",
        "json_pointer": "/pooled_all_datasets/recomputed/mean",
        "source_sha256": f6_hash,
        "ci_95": {
            "pointer_lo": "/pooled_all_datasets/recomputed/lo",
            "pointer_hi": "/pooled_all_datasets/recomputed/hi"
        },
        "baseline": 0.0,
        "denominator": "/pooled_all_datasets/recomputed/n"
    })
    
    # F6 NAIP excluded
    claims.append({
        "id": "F6.psnr.without_naip",
        "plain_text": "SR vs bicubic PSNR excluding NAIP",
        "technical_text": "Pooled SR vs bicubic PSNR excluding NAIP dataset",
        "unit": "dB",
        "evidence_label": "real",
        "f9_verdict": "keep",
        "status": "PASS",
        "source_file": "experiments/results/f6.json",
        "json_pointer": "/naip_excluded_sensitivity/recomputed/mean",
        "source_sha256": f6_hash,
        "ci_95": {
            "pointer_lo": "/naip_excluded_sensitivity/recomputed/lo",
            "pointer_hi": "/naip_excluded_sensitivity/recomputed/hi"
        },
        "baseline": 0.0,
        "denominator": "/naip_excluded_sensitivity/recomputed/n"
    })
    
    # Populate values
    for claim in claims:
        data_path = base_dir / claim['source_file']
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        claim['value'] = resolve_pointer(data, claim['json_pointer'])
        if 'ci_95' in claim and claim['ci_95']:
            claim['ci_95']['lo'] = resolve_pointer(data, claim['ci_95']['pointer_lo'])
            claim['ci_95']['hi'] = resolve_pointer(data, claim['ci_95']['pointer_hi'])
        
        if 'denominator' in claim and isinstance(claim['denominator'], str):
            try:
                claim['denominator_value'] = resolve_pointer(data, claim['denominator'])
            except:
                pass
                
    forbidden = [
        {"pattern": "1.60 km²", "reason": "F7 mapped area presented as a detection"},
        {"pattern": "1.60 km2", "reason": "F7 mapped area presented as a detection"},
        {"pattern": "89.4 % of boundary blocks differ", "reason": "presented as SR adding information"},
        {"pattern": "89.4%", "reason": "presented as SR adding information"},
        {"pattern": "block-sum PASS", "reason": "as a quality result"},
        {"pattern": "made the rule harder", "reason": "e_b escalation"},
        {"pattern": "11.8 % placebo FAR", "reason": "x3/x4/x9 number"},
        {"pattern": "11.8%", "reason": "x3/x4/x9 number"},
        {"pattern": "fine-tuning", "reason": "physical-4 GB or fine-tuning claim"}
    ]
    
    pairs = [
        ["F5.iou.with_naip", "F5.iou.without_naip"],
        ["F2.window_far"]
    ]
    
    output = {
        "claims": claims,
        "forbidden": forbidden,
        "pairs": pairs
    }
    
    out_path = base_dir / 'pitch' / 'claims.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    md_path = base_dir / 'pitch' / 'CLAIMS.md'
    with open(md_path, 'w') as f:
        f.write("# TrustSR Claims Registry\n\n")
        f.write("This file is automatically generated. Do not edit.\n\n")
        for claim in claims:
            f.write(f"## {claim['id']}\n")
            f.write(f"- **Text**: {claim['plain_text']}\n")
            f.write(f"- **Value**: {claim['value']} {claim['unit']}\n")
            f.write(f"- **Source**: `{claim['source_file']}` at `{claim['json_pointer']}`\n\n")

if __name__ == '__main__':
    main()
