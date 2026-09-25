import pytest


def test_model_artifact_mismatch_is_rejected_before_import(tmp_path):
    from risk.model import verify_artifacts
    (tmp_path / 'load.py').write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match='hash'):
        verify_artifacts(tmp_path, {'load.py': '0' * 64})


def test_label_inventory_inspection_reads_features(tmp_path):
    import sqlite3
    from risk.r4_labels import inspect_inventory
    path = tmp_path / 'labels.gpkg'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT, srs_id INTEGER, min_x REAL, min_y REAL, max_x REAL, max_y REAL)')
        db.execute("INSERT INTO gpkg_contents VALUES ('slides','features',4326,0,0,1,1)")
        db.execute('CREATE TABLE slides (fid INTEGER, event_date TEXT)')
        db.execute("INSERT INTO slides VALUES (1,'2024-07-30')")
    result = inspect_inventory(path)
    assert result['feature_count'] == 1
    assert result['event_dates'] == ['2024-07-30']
