"""One bounded GPU1 calibration. Never starts training or opens TEST."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("artifacts/large-composition-calibration-v1")
root.mkdir(exist_ok=False)
command = [
    ".venv-qwen35/bin/python",
    "-u",
    "-m",
    "cad_env.learning.extreme_benchmark",
    "--dataset",
    "artifacts/large-composition-v1",
    "--model",
    "artifacts/release-trial-v1/candidate/merged",
    "--output",
    str(root / "evaluation"),
]
frozen = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path("cad_env").rglob("*.py")
}
state = {
    "status": "running",
    "command": command,
    "started_unix": time.time(),
    "deadline_unix": time.time() + 1800,
    "physical_gpu": 1,
    "test_opened": False,
    "automatic_training": False,
    "source_sha256": frozen,
}


def persist():
    tmp = root / "supervisor.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(root / "supervisor.json")


persist()
try:
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
    )
    with (root / "run.log").open("w") as log:
        subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in frozen.items()
    ):
        raise RuntimeError("Source drift during calibration")
    state["status"] = "complete"
except Exception as exc:
    state.update(
        status="failed", error=str(exc), incomplete_run_is_not_zero_success=True
    )
    raise
finally:
    state["finished_unix"] = time.time()
    persist()
