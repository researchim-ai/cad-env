"""Surface metrics in fixed coordinates, separate from strict native success.

Not official CADBench metrics: no principal-axis alignment, approximate samples.
"""

import json
from pathlib import Path

import numpy as np


def surface_points(mesh, count=2048, seed=42):
    vertices = np.asarray(mesh["vertices"], dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh["faces"], dtype=int).reshape(-1, 3)
    triangles = vertices[faces]
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    if not len(area) or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Nonempty finite triangle surface required")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return (
        selected[:, 0]
        + uv[:, 0, None] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1, None] * (selected[:, 2] - selected[:, 0])
    )


def surface_metrics(candidate, reference, *, count=2048, seed=42):
    from scipy.spatial import cKDTree

    a = json.loads(Path(candidate).read_text())
    b = json.loads(Path(reference).read_text())
    x = surface_points(a, count, seed)
    y = surface_points(b, count, seed)
    vertices = np.asarray(b["vertices"])
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if diagonal <= 0:
        raise ValueError("Degenerate reference bounds")
    xy = cKDTree(y).query(x)[0]
    yx = cKDTree(x).query(y)[0]
    return {
        "normalized_chamfer_l1": float((xy.mean() + yx.mean()) / (2 * diagonal)),
        "surface_coverage_1pct": float(
            ((xy <= 0.01 * diagonal).mean() + (yx <= 0.01 * diagonal).mean()) / 2
        ),
        "samples_per_surface": count,
        "seed": seed,
        "alignment": "none; fixed millimeter coordinates",
        "warning": "Approximate diagnostic; finite independent surface samples have sampling error, not official CADBench SIoU/CD",
    }
