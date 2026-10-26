import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

import ppis_model
from ppis_config import CONFIG


_ROOT = Path(__file__).resolve().parent


def _safe_read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except OSError:
        return None


def sha256_file(path: os.PathLike) -> Optional[str]:
    data = _safe_read_bytes(Path(path))
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_code_fingerprints(extra_files: Optional[Iterable[os.PathLike]] = None) -> Dict[str, Dict[str, object]]:
    tracked = [
        _ROOT / "train.py",
        _ROOT / "test.py",
        _ROOT / "ppis_model.py",
        _ROOT / "ppis_config.py",
        _ROOT / "ppis_data.py",
        _ROOT / "ppis_losses.py",
        _ROOT / "ppis_repro.py",
        _ROOT / "ppis_splits.py",
    ]
    if extra_files:
        tracked.extend(Path(path) for path in extra_files)

    fingerprints = {}
    seen = set()
    for path in tracked:
        resolved = Path(path).resolve()
        key = os.path.relpath(resolved, _ROOT) if resolved.is_relative_to(_ROOT) else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        fingerprints[key] = {
            "exists": resolved.exists(),
            "sha256": sha256_file(resolved) if resolved.exists() else None,
            "size_bytes": resolved.stat().st_size if resolved.exists() else None,
            "mtime": resolved.stat().st_mtime if resolved.exists() else None,
        }
    return fingerprints


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _run_command(command: Iterable[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def collect_runtime_audit(extra_files: Optional[Iterable[os.PathLike]] = None) -> Dict[str, object]:
    config_snapshot = dict(CONFIG.__dict__)
    config_snapshot_json = json.dumps(config_snapshot, sort_keys=True, ensure_ascii=False, default=str)

    audit = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_geometric_version": _safe_import_version("torch_geometric"),
        "torch_scatter_version": _safe_import_version("torch_scatter"),
        "e3nn_version": _safe_import_version("e3nn"),
        "numpy_version": _safe_import_version("numpy"),
        "pandas_version": _safe_import_version("pandas"),
        "ipr_assignment_mode": getattr(CONFIG, "ipr_assignment_mode", None),
        "ipr_sinkhorn_iterations": getattr(CONFIG, "ipr_sinkhorn_iterations", None),
        "ipr_aux_entropy_weight": getattr(CONFIG, "ipr_aux_entropy_weight", None),
        "ipr_aux_balance_weight": getattr(CONFIG, "ipr_aux_balance_weight", None),
        "ipr_aux_compact_weight": getattr(CONFIG, "ipr_aux_compact_weight", None),
        "ipr_aux_smooth_weight": getattr(CONFIG, "ipr_aux_smooth_weight", None),
        "ppis_model_module_path": str(Path(ppis_model.__file__).resolve()),
        "ppis_model_module_sha256": sha256_file(Path(ppis_model.__file__).resolve()),
        "config_snapshot_sha256": sha256_text(config_snapshot_json),
        "env_conda_prefix": os.environ.get("CONDA_PREFIX"),
        "env_conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "pip_freeze": _run_command([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": _run_command(["nvidia-smi"]),
        "file_fingerprints": collect_code_fingerprints(extra_files=extra_files),
    }
    return audit


def write_runtime_audit_json(output_path: os.PathLike, extra_files: Optional[Iterable[os.PathLike]] = None) -> Dict[str, object]:
    audit = collect_runtime_audit(extra_files=extra_files)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return audit
