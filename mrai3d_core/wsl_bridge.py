"""
WSL2 Ubuntu Execution Bridge for mrAIGen3D.

Delegates TripoSR and TripoSplat inference to WSL2 Ubuntu using conda environment `gensfm`.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional


def windows_to_wsl_path(path: Path | str) -> str:
    """Convert Windows path (e.g. D:\\Meshroom\\...) to WSL path (/mnt/d/Meshroom/...)."""
    p_str = str(path)
    match = re.match(r"^([a-zA-Z]):[\\/](.*)", p_str)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p_str.replace("\\", "/")


def is_wsl_available() -> bool:
    try:
        res = subprocess.run(["wsl", "-l", "-q"], capture_output=True, text=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def run_ai3d_in_wsl(
    image_path: Path,
    output_obj: Path,
    output_glb: Path,
    output_ply: Path,
    model_name: str = "triposr",
    mc_resolution: int = 192,
    chunk_size: int = 4096,
    num_gaussians: int = 131072,
    distro: str = "Ubuntu",
    conda_env: str = "gensfm",
    logger: Optional[logging.Logger] = None,
) -> int:
    """Execute mrAIGen3D runner inside WSL2."""
    log = logger or logging.getLogger("mrAIGen3D.wsl_bridge")

    output_obj.parent.mkdir(parents=True, exist_ok=True)

    wsl_plugin_root = windows_to_wsl_path(Path(__file__).resolve().parents[1])
    wsl_script = f"{wsl_plugin_root}/cli/mr_ai3d_runner.py"

    cmd_args: List[str] = [
        f"--image {shlex.quote(windows_to_wsl_path(image_path.resolve()))}",
        f"--output-obj {shlex.quote(windows_to_wsl_path(output_obj.resolve()))}",
        f"--output-glb {shlex.quote(windows_to_wsl_path(output_glb.resolve()))}",
        f"--output-ply {shlex.quote(windows_to_wsl_path(output_ply.resolve()))}",
        f"--model {shlex.quote(model_name)}",
        f"--mc-resolution {int(mc_resolution)}",
        f"--chunk-size {int(chunk_size)}",
        f"--num-gaussians {int(num_gaussians)}",
    ]

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
        f"  \"$HOME/miniconda3/envs/{conda_env}/bin/python\"; do\n"
        "  if [ -x \"$candidate\" ]; then\n"
        "    PY_BIN=\"$candidate\"\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "if [ -z \"$PY_BIN\" ]; then PY_BIN=\"$(which python3 || echo python)\"; fi\n"
        f"exec \"$PY_BIN\" {shlex.quote(wsl_script)} {' '.join(cmd_args)}\n"
    )

    runner_script_win = output_obj.parent / "_mr_ai3d_wsl_runner.sh"
    runner_script_win.write_bytes(bash_commands.replace("\r\n", "\n").encode("utf-8"))
    wsl_runner_script = windows_to_wsl_path(runner_script_win)

    wsl_cmd = ["wsl", "-d", distro, "bash", wsl_runner_script]
    log.info("Executing mrAIGen3D via WSL2: %s", " ".join(wsl_cmd))

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
        log.info("[mrAIGen3D WSL] %s", line.rstrip())

    process.stdout.close()
    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(f"mrAIGen3D WSL execution failed with exit code: {returncode}")

    return returncode

