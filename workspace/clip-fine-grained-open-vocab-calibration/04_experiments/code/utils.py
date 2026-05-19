"""Utilities: seed control, logging, env capture."""
from __future__ import annotations
import os
import json
import random
import hashlib
import subprocess
import platform
import datetime as _dt
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, torch (cpu and cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "no-git"


def env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_hash": git_hash(),
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
    }
    try:
        import open_clip
        snap["open_clip"] = open_clip.__version__
    except Exception:
        pass
    try:
        import numpy as _np
        snap["numpy"] = _np.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        snap["gpu_name"] = torch.cuda.get_device_name(0)
    return snap


def file_hash(path: str | Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(n)
        h.update(chunk)
    return h.hexdigest()[:16]


def write_run_log(run_id: str, args: Dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "args": args,
        "env": env_snapshot(),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return log_path


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
