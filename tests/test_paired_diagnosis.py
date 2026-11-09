from scripts.diagnose_paired_failures import compare


def program(name="Hole", depth=4, circles=None):
    return [
        {
            "op": "sketch_circles",
            "name": name + "Sketch",
            "circles": circles or [{"x": 1, "y": 2, "radius": 3}],
        },
        {"op": "pocket", "name": name, "profile": name + "Sketch", "length": depth},
    ]


def test_matching_ignores_object_names():
    result = compare(program("HoleCB"), program("RenamedCounterbore"))
    assert result["matched_counterbores"] == 1


def test_wrong_depth_is_missing_feature():
    result = compare(program(depth=4), program(depth=8))
    assert result["matched_cuts"] == 0
    assert result["unmatched_student_cuts"] == 1


def test_unpocketed_circle_is_not_completed_cut():
    result = compare(program(), program()[:1])
    assert result["matched_cuts"] == 0


def test_grouped_circles_and_duplicate_multiplicity():
    c = {"x": 1, "y": 2, "radius": 3}
    teacher = program(circles=[c, c])
    result = compare(teacher, program())
    assert result["matched_cuts"] == 1
    assert len(result["missing"]) == 1
