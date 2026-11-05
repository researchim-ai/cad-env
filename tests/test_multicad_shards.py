import pytest

from cad_env.learning.multicad_shards import merge
from cad_env.multicad.backend import MultiCADBackend
from cad_env.multicad.dataset import audit, generate


def test_native_shards_merge_without_duplicate_tasks_or_split_reassignment(tmp_path):
    try:
        MultiCADBackend("freecad")
    except RuntimeError:
        pytest.skip("Native FreeCAD required")
    roots = []
    for count in [1, 2]:
        root = tmp_path / str(count)
        generate(
            root,
            engines=("freecad",),
            seeds=count,
            families=("stepped_bore",),
            modes=("edit",),
            modalities=("image",),
            varied_edits=True,
        )
        roots.append(root)
    output = tmp_path / "merged"
    report = merge(roots, output)
    assert report["accepted"] == 2 and len(report["source_shards"]) == 2
    assert audit(output)["ok"]
    assert len((output / "exports/train.feedback.jsonl").read_text().splitlines()) == 4
    assert len(list((output / "episodes").iterdir())) == 2
