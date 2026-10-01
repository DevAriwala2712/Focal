import json
import hashlib
from pathlib import Path
import pytest
import os

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

def test_claims_roundtrip_and_hash():
    base_dir = Path(__file__).resolve().parent.parent
    claims_path = base_dir / 'pitch' / 'claims.json'
    
    assert claims_path.exists(), "claims.json must exist"
    with open(claims_path, 'r') as f:
        registry = json.load(f)
        
    for claim in registry['claims']:
        source_path = base_dir / claim['source_file']
        assert source_path.exists(), f"Source file {source_path} does not exist"
        
        # Check hash
        current_hash = file_sha256(source_path)
        assert current_hash == claim['source_sha256'], f"STALE hash for {claim['id']}"
        
        # Round-trip value
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        actual_val = resolve_pointer(data, claim['json_pointer'])
        assert actual_val == claim['value'], f"Value mismatch for {claim['id']}"
        
        if claim.get('ci_95'):
            lo = resolve_pointer(data, claim['ci_95']['pointer_lo'])
            hi = resolve_pointer(data, claim['ci_95']['pointer_hi'])
            assert lo == claim['ci_95']['lo']
            assert hi == claim['ci_95']['hi']
            assert lo <= claim['value'] <= hi, f"Value outside CI for {claim['id']}"

def test_no_forbidden_patterns():
    base_dir = Path(__file__).resolve().parent.parent
    claims_path = base_dir / 'pitch' / 'claims.json'
    
    with open(claims_path, 'r') as f:
        registry = json.load(f)
        
    forbidden = registry.get('forbidden', [])
    search_dirs = [base_dir / 'demo', base_dir / 'pitch']
    
    for d in search_dirs:
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for file in files:
                if file == 'claims.json' or file == 'build_claims.py':
                    continue # these define the forbidden patterns, skip them
                if file.endswith(('.py', '.ts', '.md', '.tsx', '.json')):
                    filepath = Path(root) / file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for fb in forbidden:
                            assert fb['pattern'] not in content, f"Forbidden pattern '{fb['pattern']}' found in {filepath}."
