import json

import pytest

from scripts.audit_paired_drawings import audit_pairs
from scripts.generate_paired_drawings import wait_for_primary


def test_paired_audit_refuses_partial_generation_before_opening_geometry(tmp_path):
    primary, paired = tmp_path / "primary", tmp_path / "paired"
    primary.mkdir()
    paired.mkdir()
    (primary / "batch-summary.json").write_text(json.dumps({"status": "running"}))
    (paired / "summary.json").write_text(json.dumps({"status": "running"}))
    (paired / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="finish"):
        audit_pairs(primary, paired)


def test_primary_failure_is_not_replaced_with_a_different_task(tmp_path):
    (tmp_path / "batch-summary.json").write_text(
        json.dumps({"failed_shards": {"iso-000": "invalid native solid"}})
    )
    with pytest.raises(RuntimeError, match="Primary shard failed: iso-000"):
        wait_for_primary(tmp_path, {"name": "iso-000"}, 10)
