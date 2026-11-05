from cad_env.learning.data_census import graph_signature


def test_graph_signature_tracks_dependencies_but_not_dimension_or_name_changes():
    a = [
        {"op": "create_body", "name": "B"},
        {"op": "sketch_rectangle", "name": "S", "body": "B", "length": 40, "width": 30},
        {"op": "pad", "name": "P", "body": "B", "profile": "S", "length": 6},
    ]
    b = [
        {"op": "create_body", "name": "Body"},
        {
            "op": "sketch_rectangle",
            "name": "Sketch",
            "body": "Body",
            "length": 100,
            "width": 50,
        },
        {"op": "pad", "name": "Pad", "body": "Body", "profile": "Sketch", "length": 9},
    ]
    assert graph_signature(a) == graph_signature(b)
    b[-1]["profile"] = "Body"
    assert graph_signature(a) != graph_signature(b)
