"""Generate composed parts with paired draft ISO/GOST drawing inputs."""

import argparse
import json

from .__main__ import generate
from .compositions import FAMILIES, LEVELS, make_task


def factory(key, seed):
    family, level = key.split(":")
    return make_task(family, level, seed)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--per-family-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    args = p.parse_args()
    result = generate(
        args.output,
        args.per_family_level,
        args.seed,
        [f + ":" + l for f in args.families for l in args.levels],
        task_factory=factory,
        drawings=True,
    )
    print(
        json.dumps(
            {k: result[k] for k in ["attempted", "accepted", "counts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
