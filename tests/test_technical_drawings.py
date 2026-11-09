import copy
import json
import subprocess

import pytest

from cad_env.drawings.dataset import transformed_row, transformed_task
from cad_env.drawings.export import audit_drawing, export_drawing
from cad_env.learning.extreme_tasks import make_extreme_task


def test_retarget_preserves_causal_state_history_and_completion():
    from cad_env.agent.observations import agent_messages

    task = make_extreme_task("multi_support_fixture", 8, "challenge")
    history = [{"action": {"op": "create_body", "name": "Body"}, "execution_ok": True}]
    state = {"objects": [{"name": "Body", "type": "PartDesign::Body"}]}
    row = {
        "task_id": task["task_id"],
        "images": ["old.png"],
        "prompt": agent_messages(
            task,
            {"state": state, "remaining_steps": 40, "goal_images": ["old.png"]},
            history,
            visual=True,
        ),
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"op":"finish"}'}],
            }
        ],
    }
    before = copy.deepcopy(row)
    updated = transformed_task(
        task,
        "drawings/parent",
        ["drawings/parent/vlm/tile-01.png", "drawings/parent/vlm/tile-02.png"],
    )
    result = transformed_row(row, updated)
    text = result["prompt"][-1]["content"][0]["text"]
    context, _ = json.JSONDecoder().raw_decode(text)
    assert context["state"] == state
    assert context["recent_transitions"] == history
    assert context["remaining_steps"] == 40
    assert result["completion"] == before["completion"]
    assert row == before
    assert "oracle_actions" not in text
    assert result["images"] == updated["goal_images"]
    assert sum(b.get("type") == "image" for b in result["prompt"][-1]["content"]) == 2


def test_unsupported_feature_schedule_is_rejected_before_export(tmp_path):
    source = tmp_path / "source.step"
    source.write_text("not a CAD model")
    with pytest.raises(ValueError, match="complete feature-schedule adapter"):
        export_drawing(source, tmp_path / "output", task={"parameters": {"unknown": 1}})
    assert not (tmp_path / "output").exists()


@pytest.fixture(scope="module")
def native_bundle(tmp_path_factory):
    from cad_env.agent.backend import FreeCADBackend

    try:
        backend = FreeCADBackend(timeout=120)
    except RuntimeError:
        pytest.skip("Native FreeCAD required")
    tmp = tmp_path_factory.mktemp("technical-drawing")
    task = make_extreme_task("multi_support_fixture", 1, "challenge")
    result = backend.run(
        task["oracle_actions"],
        tmp / "oracle",
        reference_actions=task["reference_actions"],
        expected_metrics=task["analytic_metrics"],
        reward_mode="strict_success",
    )
    assert result["execution_ok"] and result["validation"]["task_success"]
    bundle = tmp / "drawing"
    manifest = export_drawing(tmp / "oracle/output.FCStd", bundle, task=task)
    return tmp, task, bundle, manifest


def test_native_drawing_bundle_and_pdf_page_size(native_bundle):
    _, task, bundle, manifest = native_bundle
    assert audit_drawing(bundle)["ok"]
    native = json.loads((bundle / "native.json").read_text())
    assert native["specification_verified"]
    assert native["section"]["cap_area_mm2"] > 0
    assert native["views"]["section"]["hatch_segments"]
    assert (
        native["views"]["right"]["center_mm"][0]
        < native["views"]["front"]["center_mm"][0]
    )
    assert (
        native["views"]["top"]["center_mm"][1]
        > native["views"]["front"]["center_mm"][1]
    )
    assert native["views"]["top"]["direction"] == [0, 0, 1]
    for dimension in native["dimensions"]:
        assert dimension["value_mm"] > 0
    info = subprocess.check_output(["pdfinfo", str(bundle / "drawing.pdf")], text=True)
    assert "1191 x 842 pts" in info
    assert manifest["standards_conformance"] == "not_assessed"
    assert manifest["native_spec_check_count"] > 30
    assert "BASE" in (bundle / "sheet-02.svg").read_text()
    for hole in task["parameters"]["cuts"]:
        assert any(hole["name"] in p.read_text() for p in bundle.glob("sheet-*.svg"))
    tiles = json.loads((bundle / "tiles.json").read_text())["tiles"]
    assert len(tiles) == len(manifest["vlm_tiles"])
    assert min(t["estimated_min_font_px_at_327680"] for t in tiles) >= 10.5


def test_native_rejects_wrong_depth_and_missing_feature(native_bundle):
    tmp, task, _, _ = native_bundle
    wrong = copy.deepcopy(task)
    wrong["parameters"]["cuts"][0]["depth"] += 1
    with pytest.raises(RuntimeError, match="no drawing result"):
        export_drawing(tmp / "oracle/output.FCStd", tmp / "wrong-depth", task=wrong)
    assert "disagrees with CAD" in (tmp / "wrong-depth/native.log").read_text()
    missing = copy.deepcopy(task)
    missing["parameters"]["cuts"].pop()
    with pytest.raises(RuntimeError, match="no drawing result"):
        export_drawing(
            tmp / "oracle/output.FCStd", tmp / "missing-feature", task=missing
        )
    assert (
        "cover every native pad and pocket"
        in (tmp / "missing-feature/native.log").read_text()
    )


def test_bundle_tampering_rejected(native_bundle):
    _, _, bundle, _ = native_bundle
    target = bundle / "vlm/tile-01.png"
    original = target.read_bytes()
    try:
        target.write_bytes(b"changed")
        report = audit_drawing(bundle)
        assert not report["ok"]
        assert any("tile-01.png" in error for error in report["errors"])
    finally:
        target.write_bytes(original)


def test_test_geometry_is_not_opened_by_development_conversion(tmp_path, monkeypatch):
    import cad_env.drawings.dataset as module

    source = tmp_path / "source"
    source.mkdir()
    (source / "summary.json").write_text("{}")
    (source / "manifest.jsonl").write_text(
        '{"task_id":"sealed","split":"test","accepted":true}\n'
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("General native audit must not open sealed TEST geometry")

    monkeypatch.setattr(module, "audit_dataset", forbidden)
    with pytest.raises(ValueError, match="TEST geometry is not opened"):
        module.convert_dataset(source, tmp_path / "output")
    assert not module.audit_technical_dataset(source)["ok"]


@pytest.fixture(scope="module")
def converted_small_corpus(tmp_path_factory):
    from cad_env.drawings.dataset import convert_dataset
    from cad_env.learning.factory import generate_dataset

    tmp = tmp_path_factory.mktemp("drawing-corpus")
    selected = {}
    for seed in range(100):
        task = make_extreme_task("heterogeneous_plate", seed, "challenge")
        if task["split"] in ("train", "validation"):
            selected.setdefault(task["split"], task)
        if len(selected) == 2:
            break
    generate_dataset(
        tmp / "source",
        task_specs=list(selected.values()),
        families=("heterogeneous_plate",),
        modes=("create",),
        preferences=False,
        workers=2,
        timeout=120,
    )
    convert_dataset(tmp / "source", tmp / "converted")
    return tmp / "converted"


def test_corpus_rejects_rehashed_label_and_future_state_corruption(
    converted_small_corpus,
):
    from cad_env.drawings.dataset import audit_technical_dataset
    from cad_env.drawings.export import sha

    root = converted_small_corpus
    assert audit_technical_dataset(root)["ok"]
    target = root / "exports/train.feedback.jsonl"
    summary_path = root / "summary.json"
    raw = target.read_bytes()
    original_summary = summary_path.read_bytes()
    try:
        for mode in ("completion", "future_state"):
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            if mode == "completion":
                rows[0]["completion"][0]["content"][0]["text"] = '{"op":"finish"}'
            else:
                text = rows[0]["prompt"][-1]["content"][0]["text"]
                context, end = json.JSONDecoder().raw_decode(text)
                context["state"]["future_target"] = "answer from a later step"
                rows[0]["prompt"][-1]["content"][0]["text"] = (
                    json.dumps(context) + text[end:]
                )
            target.write_text("".join(json.dumps(r) + "\n" for r in rows))
            summary = json.loads(original_summary)
            summary["export_sha256"]["exports/train.feedback.jsonl"] = sha(target)
            summary["artifact_sha256"]["exports/train.feedback.jsonl"] = sha(target)
            summary_path.write_text(json.dumps(summary))
            report = audit_technical_dataset(root)
            assert not report["ok"]
            assert any(
                "Causal prompt/label/image conversion" in e for e in report["errors"]
            )
    finally:
        target.write_bytes(raw)
        summary_path.write_bytes(original_summary)


def test_standard_profiles_have_distinct_frames_and_physical_lettering():
    import re
    import xml.etree.ElementTree as ET

    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    from cad_env.drawings.lettering import FONT, outline_text
    from cad_env.drawings.profiles import ACTIVE, METADATA, get_profile
    from cad_env.drawings.sheets import document, frame, text

    # Measure a real outline, independently of the SVG's declared height attribute.
    font = TTFont(FONT)
    glyphs = font.getGlyphSet()
    pen = BoundsPen(glyphs)
    glyphs["H"].draw(pen)
    element = ET.fromstring(outline_text(0, 0, "H", 3.5))
    scale = float(re.search(r"scale\((\S+) ", element.get("transform"))[1])
    assert (pen.bounds[3] - pen.bounds[1]) * scale == pytest.approx(3.5)
    assert 'data-text="АБВЁё⌀№"' in outline_text(0, 0, "АБВЁё⌀№", 2.5)
    with pytest.raises(ValueError, match="lacks character"):
        outline_text(0, 0, "\U0001f680", 2.5)
    for standard, margins, block, width, heights in (
        ("iso", ("20", "10", "390", "277"), "iso-title-block", "180", ("50", "50")),
        ("gost", ("20", "5", "395", "287"), "gost-form-", "185", ("55", "15")),
    ):
        token = ACTIVE.set(get_profile(standard))
        metadata = METADATA.set({"designation": "CAD.001"})
        try:
            for page in (1, 2):
                svg = ET.fromstring(
                    document(frame("fallback", "heterogeneous_plate", page, 2))
                )
                elements = {e.get("id"): e for e in svg.iter() if e.get("id")}
                border = elements["standard-border"]
                assert (
                    tuple(border.get(k) for k in ("x", "y", "width", "height"))
                    == margins
                )
                title = elements[
                    block if standard == "iso" else block + ("1" if page == 1 else "2a")
                ]
                assert (title.get("width"), title.get("height")) == (
                    width,
                    heights[page - 1],
                )
                if standard == "gost":
                    assert (
                        elements["gost-field-26"].get("transform")
                        == "translate(90 19) rotate(180)"
                    )
                    assert "gost-inventory-strip" in elements
                    assert 'data-text="⌀12,5"' in text(0, 0, "D12.5")
                else:
                    assert 'data-text="⌀12.5"' in text(0, 0, "D12.5")
                assert "<text " not in ET.tostring(svg, encoding="unicode")
        finally:
            ACTIVE.reset(token)
            METADATA.reset(metadata)


@pytest.mark.parametrize("standard", ["iso", "gost"])
def test_native_standard_export_and_rehashed_report_tampering(native_bundle, standard):
    from cad_env.drawings.export import sha

    tmp, task, _, _ = native_bundle
    output = tmp / standard
    manifest = export_drawing(
        tmp / "oracle/output.FCStd",
        output,
        task=task,
        standard=standard,
        metadata={"designation": "CAD.002", "synthetic_example": True},
    )
    assert audit_drawing(output)["ok"]
    assert not list(output.rglob("*.ttf"))
    assert not list(output.rglob("*.sfd"))
    assert "font-provenance.json" in manifest["files"]
    font_info = json.loads((output / "font-provenance.json").read_text())
    assert not font_info["font_binary_bundled"]
    report_path = output / "conformance.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "nonconforming"
    assert not report["can_claim_full_conformance"]
    assert report["pending"]
    assert all(
        c["status"] == "pass" for c in report["checks"] if c["id"].startswith("page_")
    )
    assert json.loads((output / "native.json").read_text())["dimension_offset_mm"] >= 10
    info = subprocess.check_output(["pdfinfo", str(output / "drawing.pdf")], text=True)
    assert "1191 x 842 pts" in info
    # A caller cannot turn an incomplete review into a declaration by rehashing JSON.
    report["can_claim_full_conformance"] = True
    report_path.write_text(json.dumps(report))
    manifest["files"]["conformance.json"] = sha(report_path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    assert not audit_drawing(output)["ok"]


def test_strict_conformance_gate_refuses_unreviewed_release(native_bundle):
    tmp, task, _, _ = native_bundle
    output = tmp / "strict-gost"
    with pytest.raises(ValueError, match="conformance"):
        export_drawing(
            tmp / "oracle/output.FCStd",
            output,
            task=task,
            standard="gost",
            require_conformance=True,
        )
    assert (output / "drawing.pdf").is_file()
    report = json.loads((output / "conformance.json").read_text())
    assert not report["can_claim_full_conformance"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_metadata_rejects_silent_overflow_and_preserves_null_as_blank():
    from cad_env.drawings.profiles import (
        ACTIVE,
        METADATA,
        get_profile,
        validate_metadata,
    )
    from cad_env.drawings.sheets import frame

    assert validate_metadata({"creator": None}) == {}
    with pytest.raises(ValueError, match="strings"):
        validate_metadata({"issue_date": 20261109})
    with pytest.raises(ValueError, match="Unknown"):
        validate_metadata({"signed": "yes"})
    token = ACTIVE.set(get_profile("iso"))
    metadata = METADATA.set({"owner": "OWNER " * 100})
    try:
        with pytest.raises(ValueError, match="exceeds"):
            frame("CAD.001", "heterogeneous_plate", 1, 2)
    finally:
        ACTIVE.reset(token)
        METADATA.reset(metadata)


@pytest.mark.parametrize("standard", ["iso", "gost"])
def test_standard_corpus_conversion_preserves_labels_and_parent_splits(
    converted_small_corpus, standard
):
    from cad_env.drawings.dataset import audit_technical_dataset, convert_dataset

    source = converted_small_corpus.parent / "source"
    output = converted_small_corpus.parent / ("corpus-" + standard)
    summary = convert_dataset(
        source, output, standard=standard, metadata={"synthetic_example": True}
    )
    assert summary["drawing_standard"] == standard
    assert summary["standards_conformance"] == "nonconforming"
    assert audit_technical_dataset(output)["ok"]
    for split in ("train", "validation"):
        original = [
            json.loads(s)
            for s in (source / "exports" / (split + ".feedback.jsonl"))
            .read_text()
            .splitlines()
        ]
        converted = [
            json.loads(s)
            for s in (output / "exports" / (split + ".feedback.jsonl"))
            .read_text()
            .splitlines()
        ]
        assert [r["completion"] for r in original] == [
            r["completion"] for r in converted
        ]
        assert [r["task_id"] for r in original] == [r["task_id"] for r in converted]
