"""Independent native closed-mesh grading, avoiding Blender coplanar boolean bugs."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import manifold3d
import numpy as np


def compare(candidate, reference, tolerance=1e-5):
    def load(path):
        data = json.loads(Path(path).read_text())
        vertices = np.asarray(data["vertices"], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(data["faces"], dtype=np.uint32).reshape(-1, 3)
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise ValueError("Nonempty finite mesh required")
        shape = manifold3d.Manifold(
            manifold3d.Mesh(vert_properties=vertices, tri_verts=faces)
        )
        if shape.status() != manifold3d.Error.NoError or shape.volume() <= 0:
            raise ValueError(
                "Expected oriented closed manifold mesh: " + str(shape.status())
            )
        bounds = np.concatenate([vertices.min(axis=0), vertices.max(axis=0)])
        return shape, bounds

    a, ab = load(candidate)
    b, bb = load(reference)
    delta = max(0.0, (a - b).volume()) + max(0.0, (b - a).volume())
    scale = max(b.volume(), 1.0)
    bounds_error = float(np.abs(ab - bb).max())
    union = (a.volume() + b.volume() + delta) / 2
    return {
        "geometry_matches": bool(
            delta / scale <= tolerance and bounds_error <= tolerance
        ),
        "symmetric_difference_volume": delta,
        "relative_symmetric_difference": delta / scale,
        "bounds_error": bounds_error,
        "native_volume_iou": max(0.0, (a.volume() + b.volume() - delta) / 2) / union,
        "grading_runtime": {
            "engine": "manifold3d",
            "version": importlib.metadata.version("manifold3d"),
            "input": "actual Blender mesh; fixed mm coordinates",
        },
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tolerance", type=float, default=1e-5)
    a = p.parse_args()
    Path(a.output).write_text(
        json.dumps(compare(a.candidate, a.reference, a.tolerance), indent=2)
    )


if __name__ == "__main__":
    main()
