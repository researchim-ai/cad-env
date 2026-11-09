"""Process-isolated native executors; missing engines never produce fake labels."""

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cad_env.agent.backend import FreeCADBackend
from cad_env.agent.schema import validate_program

from .plan import resolve


class MultiCADBackend:
    def __init__(self, engine="freecad", *, python=None, timeout=90):
        if engine not in {"freecad", "cadquery"}:
            raise ValueError("Choose freecad or cadquery")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite timeout required")
        self.engine, self.python, self.timeout = (
            engine,
            python or sys.executable,
            timeout,
        )
        self.native = FreeCADBackend(timeout=timeout) if engine == "freecad" else None
        self.sources = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }

    def run(
        self,
        actions,
        output_dir,
        *,
        initial_actions=None,
        reference_actions=None,
        expected_metrics=None,
        tolerance=1e-05,
        mesh=True,
        record=False,
    ):
        if any(
            (
                hashlib.sha256(p.read_bytes()).hexdigest() != digest
                for p, digest in self.sources.items()
            )
        ):
            raise RuntimeError(
                "Multi-engine source changed; restart with a frozen implementation"
            )
        if not math.isfinite(tolerance) or not 0 < tolerance <= 0.01:
            raise ValueError("Tolerance must be positive and at most .01")
        initial = validate_program(initial_actions or [])
        actions = validate_program(actions)
        reference = (
            validate_program(reference_actions)
            if reference_actions is not None
            else None
        )
        resolve(initial + actions)
        if reference is not None:
            resolve(reference)
        out = Path(output_dir).resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a fresh execution directory")
        if self.native:
            result = self.native.run(
                actions,
                out,
                initial_actions=initial,
                reference_actions=reference,
                expected_metrics=expected_metrics,
                tolerance=tolerance,
                mesh=mesh,
                record=record,
                reward_mode="strict_success",
            )
            finished = bool(actions and actions[-1]["op"] == "finish")
            validation = result.setdefault("validation", {})
            validation["finished"] = finished
            validation["task_success"] = bool(
                validation.get("task_success") and finished
            )
            result["reward"] = float(validation["task_success"])
            (out / "result.json").write_text(json.dumps(result, indent=2))
            return result
        out.mkdir(parents=True)
        request = {
            "engine": self.engine,
            "actions": initial + actions,
            "reference_actions": reference,
            "expected_metrics": expected_metrics,
            "tolerance": tolerance,
            "mesh": mesh,
            "record": record,
        }
        req = out / "request.json"
        req.write_text(json.dumps(request, allow_nan=False))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "2"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        for key in ["XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"]:
            env[key] = str(out / ".runtime" / key)
        start = time.monotonic()
        command = [self.python, "-m", "cad_env.multicad.worker", str(req)]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        timeout = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (out / "stdout.log").write_bytes(stdout)
        (out / "stderr.log").write_bytes(stderr)
        result_path = out / "result.json"
        if timeout or process.returncode or (not result_path.exists()):
            result = {
                "execution_ok": False,
                "reward": 0.0,
                "error_type": "Timeout" if timeout else "WorkerCrash",
                "error": stderr.decode(errors="replace")[-2000:],
            }
        else:
            result = json.loads(result_path.read_text())
        result.update(
            backend=self.engine,
            output_dir=str(out),
            elapsed_seconds=time.monotonic() - start,
            worker_sha256=hashlib.sha256(
                Path(__file__).with_name("worker.py").read_bytes()
            ).hexdigest(),
        )
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
