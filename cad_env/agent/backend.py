"""Process-isolated execution of a restricted CAD action language."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .schema import validate_program


class FreeCADBackend:
    def __init__(self, executable: str | None = None, timeout: float = 30):
        configured = executable or os.environ.get("FREECAD_CMD")
        candidates = (
            [configured]
            if configured
            else ["FreeCADCmd", "freecadcmd", "freecadcmd-daily"]
        )
        self.executable = next(
            (shutil.which(p) for p in candidates if p and shutil.which(p)), None
        )
        if not self.executable:
            raise RuntimeError(
                "Real FreeCAD console not found. Set FREECAD_CMD; simulation is never used for labels."
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._code_fingerprints = {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("worker.py", "schema.py")
        }
        self._runtime_signature = None

    def run(
        self,
        actions,
        output_dir,
        *,
        initial_actions=None,
        reference_actions=None,
        record=False,
        mesh=False,
        tolerance=1e-5,
        expected_metrics=None,
        allowed_edits=None,
    ):
        for name, digest in self._code_fingerprints.items():
            if (
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                != digest
            ):
                raise RuntimeError(
                    "Native worker/schema changed during the run; restart with a frozen source snapshot"
                )
        # Reject arbitrary code/paths before starting the native process.
        actions = validate_program(actions)
        initial_actions = validate_program(initial_actions or [])
        if reference_actions is not None:
            reference_actions = validate_program(reference_actions)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        result_path.unlink(missing_ok=True)
        request = {
            "actions": actions,
            "initial_actions": initial_actions,
            "reference_actions": reference_actions,
            "record": record,
            "mesh": mesh,
            "tolerance": tolerance,
            "expected_metrics": expected_metrics,
            "allowed_edits": allowed_edits,
        }
        request_path = output / "request.json"
        request_path.write_text(json.dumps(request, allow_nan=False))
        worker = Path(__file__).with_name("worker.py")
        bootstrap = output / "run_worker.py"
        bootstrap.write_text(
            "import runpy\nworker = runpy.run_path("
            + repr(str(worker))
            + ')\nworker["main"]('
            + repr(str(request_path))
            + ", "
            + repr(str(result_path))
            + ")\n"
        )
        env = os.environ.copy()
        # FreeCAD must not write settings/caches to the user's normal profile.
        for key, directory in [
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_DATA_HOME", "data"),
        ]:
            path = output / ".runtime" / directory
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        started = time.monotonic()
        process = subprocess.Popen(
            [self.executable, str(bootstrap)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (output / "stdout.log").write_bytes(stdout)
        (output / "stderr.log").write_bytes(stderr)
        if timed_out:
            result = {
                "execution_ok": False,
                "error_type": "Timeout",
                "error": f"FreeCAD exceeded {self.timeout}s",
            }
        elif not result_path.is_file():
            result = {
                "execution_ok": False,
                "error_type": "WorkerCrash",
                "error": "Worker did not produce a structured result",
            }
        else:
            try:
                result = json.loads(result_path.read_text())
            except (ValueError, OSError):
                result = {
                    "execution_ok": False,
                    "error_type": "ProtocolError",
                    "error": "Invalid worker result",
                }
            if process.returncode:
                result.update(
                    execution_ok=False,
                    error_type="WorkerCrash",
                    error=f"FreeCAD exit code {process.returncode}",
                )
        if result.get("runtime"):
            signature = json.dumps(result["runtime"], sort_keys=True)
            if (
                self._runtime_signature is not None
                and signature != self._runtime_signature
            ):
                raise RuntimeError("FreeCAD/OCCT runtime changed during the run")
            self._runtime_signature = signature
        result["schema_sha256"] = self._code_fingerprints["schema.py"]
        result["execution_seconds"] = time.monotonic() - started
        result["return_code"] = process.returncode
        result["output_dir"] = str(output)
        result["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
        if result.get("artifact") and (output / result["artifact"]).is_file():
            result["artifact_sha256"] = hashlib.sha256(
                (output / result["artifact"]).read_bytes()
            ).hexdigest()
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        )
        return result
