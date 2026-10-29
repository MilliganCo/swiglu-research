from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers


def command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    files += [root / "pyproject.toml", root / "conftest.py"]
    if (root / "requirements-lock.txt").exists():
        files.append(root / "requirements-lock.txt")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot_source(root: Path, directory: Path) -> str:
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    files += sorted((root / "scripts").glob("*.*"))
    files += sorted((root / "configs").glob("*.yaml")) + sorted((root / "configs").glob("*.json"))
    files += [root / name for name in ("pyproject.toml", "conftest.py", "requirements-lock.txt", "README.md")]
    path = directory / "source_snapshot.zip"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(root).as_posix())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(root: Path) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    commit = command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = command(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "git_commit_unavailable_reason": None if commit else "directory has no Git commit",
        "git_dirty": bool(status) if commit else None, "source_sha256": source_hash(root),
        "python": platform.python_version(), "platform": platform.platform(),
        "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
        "wsl": "microsoft" in platform.release().lower(), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "cuda_runtime": torch.version.cuda,
        "nvidia_driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu": gpu.name if gpu else None,
        "compute_capability": [gpu.major, gpu.minor] if gpu else None,
        "gpu_total_bytes": gpu.total_memory if gpu else None,
        "gpu_free_bytes_at_start": torch.cuda.mem_get_info()[0] if gpu else None,
        "system_memory": command(["free", "-b"]),
        "tf32": False, "deterministic_algorithms": True,
    }


class PowerSampler:
    """Optional total-board-power samples, including other desktop GPU users."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop.is_set():
            raw = command(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
            try:
                watts = float(raw)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                break
            self.samples.append((time.perf_counter(), watts))
            self.stop.wait(0.2)

    def finish(self, tokens: int) -> dict[str, Any]:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=11)
        energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                     for a, b in zip(self.samples, self.samples[1:])) if len(self.samples) > 1 else None
        return {"power_samples": self.samples, "energy_joules": energy,
                "energy_per_token_joules": energy / tokens if energy is not None else None,
                "mean_power_watts": float(np.mean([s[1] for s in self.samples])) if self.samples else None,
                "power_scope": "total GPU board; coarse samples; not isolated process energy",
                "power_unavailable_reason": None if energy is not None else "driver supplied no usable power series"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    import fcntl
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)
