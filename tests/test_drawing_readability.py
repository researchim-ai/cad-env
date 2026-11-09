from cad_env.drawings.readability import grade_response


def test_strict_format_and_semantics_are_separate():
    target = {"x": -2.2, "counterbore_depth": None}
    strict = grade_response('{"x":-2.2,"counterbore_depth":null}', target)
    assert strict["strict_exact"] and strict["field_exact_diagnostic"]
    recovered = grade_response("{'x': -2.2, 'counterbore_depth': null}", target)
    assert recovered["field_exact_diagnostic"]
    assert not recovered["strict_json"] and not recovered["strict_exact"]


def test_recovery_cannot_execute_code_or_hide_invalid_values():
    for response in [
        "{'x':__import__('os').getcwd()}",
        '{"x":NaN}',
        '{"x":true}',
        '{"x":2,"x":1}',
    ]:
        result = grade_response(response, {"x": 1})
        assert not result["strict_exact"]
        assert not result["field_exact_diagnostic"]
