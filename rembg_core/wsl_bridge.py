"""
WSL2 Ubuntu Execution Bridge for mrRemBG.

Allows seamless delegation of image matting to WSL2 where CUDA, PyTorch,
and rembg are fully configured in the `gensfm` conda environment.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence


def windows_to_wsl_path(path: Path | str) -> str:
    """Convert Windows path (e.g. D:\\Meshroom\\...) to WSL path (/mnt/d/Meshroom/...)."""
    p_str = str(path)
    # Check for drive letter e.g. C: or D:
    match = re.match(r"^([a-zA-Z]):[\\/](.*)", p_str)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p_str.replace("\\", "/")


def is_wsl_available() -> bool:
    """Check if WSL is installed and functional."""
    try:
        res = subprocess.run(["wsl", "-l", "-q"], capture_output=True, text=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def run_rembg_in_wsl(
    input_paths: Sequence[Path],
    output_rgba_dir: Path,
    output_mask_dir: Path,
    output_sfm_path: Optional[Path] = None,
    original_sfm_path: Optional[Path] = None,
    model_name: str = "u2net",
    alpha_matting: bool = False,
    af_thresh: int = 240,
    ab_thresh: int = 10,
    erode_size: int = 10,
    pad_to_square: bool = False,
    distro: str = "Ubuntu",
    conda_env: str = "gensfm",
    logger: Optional[logging.Logger] = None,
) -> int:
    """Execute mrRemBG runner inside WSL2 using the active conda environment."""
    log = logger or logging.getLogger("mrRemBG.wsl_bridge")

    output_rgba_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    wsl_plugin_root = windows_to_wsl_path(Path(__file__).resolve().parents[1])
    wsl_script = f"{wsl_plugin_root}/cli/mr_rembg_runner.py"

    cmd_args: List[str] = [
        f"--model {shlex.quote(model_name)}",
        f"--output-rgba-dir {shlex.quote(windows_to_wsl_path(output_rgba_dir))}",
        f"--output-mask-dir {shlex.quote(windows_to_wsl_path(output_mask_dir))}",
        f"--af-thresh {int(af_thresh)}",
        f"--ab-thresh {int(ab_thresh)}",
        f"--erode-size {int(erode_size)}",
    ]
    if alpha_matting:
        cmd_args.append("--alpha-matting")
    if pad_to_square:
        cmd_args.append("--pad-to-square")
    if output_sfm_path:
        cmd_args.append(f"--output-sfm {shlex.quote(windows_to_wsl_path(output_sfm_path))}")
    if original_sfm_path and original_sfm_path.is_file():
        cmd_args.append(f"--original-sfm {shlex.quote(windows_to_wsl_path(original_sfm_path))}")

    wsl_inputs = " ".join(shlex.quote(windows_to_wsl_path(p.resolve())) for p in input_paths)
    cmd_args.append(f"--images {wsl_inputs}")

    # Build bash runner script to resolve conda and python
    bash_commands = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "for cpath in \"$HOME/miniforge3\" \"$HOME/mambaforge\" \"$HOME/miniconda3\" \"$HOME/anaconda3\" \"/opt/conda\"; do\n"
        "  if [ -f \"$cpath/etc/profile.d/conda.sh\" ]; then\n"
        "    . \"$cpath/etc/profile.d/conda.sh\"\n"
        f"    conda activate {shlex.quote(conda_env)} 2>/dev/null || true\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "\n"
        "PY_BIN=\"\"\n"
        f"for candidate in \\\n"
        f"  \"$HOME/miniforge3/envs/{conda_env}/bin/python\" \\\n"
        f"  \"$HOME/miniforge3/envs/{conda_env}/bin/python3\" \\\n"
        f"  \"$HOME/miniconda3/envs/{conda_env}/bin/python\" \\\n"
        f"  \"$HOME/mambaforge/envs/{conda_env}/bin/python\"; do\n"
        "  if [ -x \"$candidate\" ]; then\n"
        "    PY_BIN=\"$candidate\"\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "if [ -z \"$PY_BIN\" ]; then PY_BIN=\"$(which python3 || echo python)\"; fi\n"
        f"exec \"$PY_BIN\" {shlex.quote(wsl_script)} {' '.join(cmd_args)}\n"
    )

    runner_script_win = output_rgba_dir.parent / "_mr_rembg_wsl_runner.sh"
    runner_script_win.write_bytes(bash_commands.replace("\r\n", "\n").encode("utf-8"))
    wsl_runner_script = windows_to_wsl_path(runner_script_win)

    wsl_cmd = ["wsl", "-d", distro, "bash", wsl_runner_script]
    log.info("Executing RemBG via WSL2: %s", " ".join(wsl_cmd))

    process = subprocess.Popen(
        wsl_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        log.info("[mrRemBG WSL] %s", line.rstrip())

    process.stdout.close()
    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(f"mrRemBG WSL runner failed with exit code: {returncode}")

    return returncode

