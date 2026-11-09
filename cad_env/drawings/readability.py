"""Separate drawing-value extraction from strict JSON compliance in diagnostics.

Literal recovery never executes code and must not be used to change CAD rewards
or silently accept invalid agent actions. Raw model responses remain the evidence.
"""

import ast
import json
import math


def grade_response(response, expected):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON number: " + value)

    strict = False
    parsed = None
    method = "invalid"
    try:
        parsed = json.loads(
            response, object_pairs_hook=pairs, parse_constant=invalid_constant
        )
        strict = True
        method = "json"
    except (ValueError, TypeError):

        def literal(node):
            if isinstance(node, ast.Dict):
                return pairs(
                    [
                        (literal(k), literal(v))
                        for k, v in zip(node.keys, node.values, strict=True)
                    ]
                )
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (str, int, float, bool, type(None))
            ):
                return node.value
            if isinstance(node, ast.Name) and node.id in ("null", "true", "false"):
                return {"null": None, "true": True, "false": False}[node.id]
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in (int, float)
            ):
                return -node.operand.value
            raise ValueError("Not a plain literal dictionary")

        try:
            parsed = literal(ast.parse(response.strip(), mode="eval").body)
            method = "literal_diagnostic_only"
        except (ValueError, SyntaxError, TypeError):
            pass
    exact = isinstance(parsed, dict) and set(parsed) == set(expected)
    if exact:
        exact = all(
            parsed[key] is None
            if value is None
            else type(parsed[key]) in (int, float)
            and math.isfinite(parsed[key])
            and math.isclose(parsed[key], value, abs_tol=1e-6, rel_tol=1e-9)
            for key, value in expected.items()
        )
    return {
        "strict_json": strict,
        "strict_exact": strict and exact,
        "field_exact_diagnostic": bool(exact),
        "parse_method": method,
    }
