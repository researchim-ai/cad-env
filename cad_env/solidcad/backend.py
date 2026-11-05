"""Native process runner for the advanced solid-graph contract."""

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend

from .schema import validate


class SolidBackend:
    def __init__(self, engine, *, python=None, blender=None, timeout=90):
        if (
            engine not in {"freecad", "cadquery", "blender"}
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid native engine/timeout")
        self.engine = engine
        self.python = python or sys.executable
        self.blender = (
            blender or os.environ.get("BLENDER_BIN") or shutil.which("blender")
        )
        self.timeout = timeout
        self.freecad = FreeCADBackend().executable if engine == "freecad" else None
        self.sources = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [
                Path(__file__).parent,
                Path(__file__).parents[1] / "multicad",
            ]
            for p in folder.glob("*.py")
        }

    def run(
        self, program, output, *, reference=None, expected_volume=None, tolerance=1e-5
    ):
        if any(
            hashlib.sha256(p.read_bytes()).hexdigest() != h
            for p, h in self.sources.items()
        ):
            raise RuntimeError("Solid engine source changed")
        program = validate(program)
        reference = validate(reference) if reference is not None else None
        if not math.isfinite(tolerance) or not 0 < tolerance <= 0.01:
            raise ValueError("Invalid tolerance")
        if expected_volume is not None and (
            not math.isfinite(expected_volume) or expected_volume <= 0
        ):
            raise ValueError("Invalid analytic volume")
        out = Path(output).resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a fresh solid output")
        out.mkdir(parents=True)
        req = out / "request.json"
        req.write_text(
            json.dumps(
                {
                    "engine": self.engine,
                    "program": program,
                    "reference": reference,
                    "expected_volume": expected_volume,
                    "tolerance": tolerance,
                },
                allow_nan=False,
            )
        )
        package_root = Path(__file__).resolve().parents[2]
        bootstrap = out / "run_solid.py"
        bootstrap.write_text(
            "import sys,runpy\nsys.path.insert(0,"
            + repr(str(package_root))
            + ')\nsys.argv=["worker",'
            + repr(str(req))
            + ']\nrunpy.run_module("cad_env.solidcad.worker",run_name="__main__")\n'
        )
        if self.engine == "freecad":
            cmd = [self.freecad, str(bootstrap)]
        elif self.engine == "blender" and self.blender:
            cmd = [
                self.blender,
                "--background",
                "--factory-startup",
                "--threads",
                "2",
                "--python-exit-code",
                "1",
                "--python",
                str(bootstrap),
            ]
        else:
            cmd = [self.python, str(bootstrap)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "2"
        for name in ["XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"]:
            env[name] = str(out / ".runtime" / name)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        timed = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (out / "stdout.log").write_bytes(stdout)
        (out / "stderr.log").write_bytes(stderr)
        path = out / "result.json"
        if timed or process.returncode or not path.exists():
            result = {
                "execution_ok": False,
                "reward": 0.0,
                "error_type": "Timeout" if timed else "WorkerCrash",
                "error": stderr.decode(errors="replace")[-1000:],
            }
        else:
            result = json.loads(path.read_text())
        result.update(
            backend=self.engine,
            output_dir=str(out),
            source_hashes={
                str(p.relative_to(package_root)): h for p, h in self.sources.items()
            },
        )
        path.write_text(json.dumps(result, indent=2))
        return result
