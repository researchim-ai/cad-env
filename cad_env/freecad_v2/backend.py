"""Isolated CPU-only execution; hashes freeze v2 sources within a backend run."""

import hashlib
import json
import math
import os
import signal
import subprocess
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend

from .schema import validate


class Backend:
    def __init__(self, timeout=90):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite timeout required")
        self.executable = FreeCADBackend().executable
        self.timeout = timeout
        self.sources = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }
        for relative in (
            "solidcad/freecad_engine.py",
            "agent/schema.py",
            "agent/backend.py",
        ):
            dependency = Path(__file__).parents[1] / relative
            self.sources[str(dependency)] = hashlib.sha256(
                dependency.read_bytes()
            ).hexdigest()

    def run(self, actions, output, *, reference=None, expected_volume=None):
        actions = validate(actions, terminal=False)
        reference = validate(reference) if reference is not None else None
        if expected_volume is not None and (
            not math.isfinite(expected_volume) or expected_volume <= 0
        ):
            raise ValueError("Invalid analytic volume")
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in self.sources.items()
        ):
            raise RuntimeError("v2 source changed during run")
        out = Path(output).absolute()
        out.mkdir(parents=True, exist_ok=False)
        request = out / "request.json"
        request.write_text(
            json.dumps(
                {
                    "actions": actions,
                    "reference": reference,
                    "expected_volume": expected_volume,
                },
                allow_nan=False,
            )
        )
        bootstrap = out / "run.py"
        bootstrap.write_text(
            "import sys,runpy\nsys.path.insert(0,"
            + repr(str(Path(__file__).absolute().parents[2]))
            + ')\nsys.argv=["worker",'
            + repr(str(request))
            + ']\nrunpy.run_module("cad_env.freecad_v2.worker",run_name="__main__")\n'
        )
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
        for key in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
            env[key] = str(out / ".runtime" / key)
        process = subprocess.Popen(
            [self.executable, str(bootstrap)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (out / "stdout.log").write_bytes(stdout)
        (out / "stderr.log").write_bytes(stderr)
        path = out / "result.json"
        if timed_out or process.returncode or not path.exists():
            result = {
                "execution_ok": False,
                "reward": 0.0,
                "error_type": "Timeout" if timed_out else "WorkerCrash",
                "error": stderr.decode(errors="replace")[-1500:],
            }
        else:
            result = json.loads(path.read_text())
        result.update(output_dir=str(out), source_hashes=self.sources)
        path.write_text(json.dumps(result, indent=2))
        return result
