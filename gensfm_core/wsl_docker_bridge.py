"""
WSL2 Ubuntu GPU Bridge, Docker GPU Runner & Low-VRAM Sequential Offloading for mrGenSfM.

Enables Windows hosts to execute the Linux/CUDA-native `dotb1ue/GenSfm` pipeline seamlessly
via WSL2 Ubuntu or Docker Desktop (WSL2 backend), with automatic Windows <-> WSL2 path translation
and Sequential Stage 1 / Stage 2 CPU-GPU offloading for 4GB-8GB VRAM GPUs.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence


LOW_VRAM_WRAPPER_SCRIPT = '''"""Auto-generated Low-VRAM Sequential Stage Offloading Wrapper for dotb1ue/GenSfm."""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("ATTN_BACKEND_WITH_ATTN_BIAS", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gensfm-root", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=str, default="dotblue/GenSfm")
    parser.add_argument("--stage1", type=str, default="")
    parser.add_argument("--stage2", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--remove-background", action="store_true")
    args = parser.parse_args()

    gensfm_root = args.gensfm_root.resolve()
    if str(gensfm_root) not in sys.path:
        sys.path.insert(0, str(gensfm_root))

    import random
    import numpy as np
    import torch
    import example_multi_image as emi
    from trellis.pipelines.gensfm import GenSfMPipeline, InputViews
    from trellis import evaluators

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available inside the execution environment.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    paths = [p.resolve() for p in args.images]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage1 and args.stage2:
        model_cfgs, options, crop_ratio = emi.inference_config(
            args.stage1, args.stage2, len(paths), args.steps if args.steps is not None else 25
        )
    else:
        pipeline = GenSfMPipeline.from_pretrained(args.pretrained).cuda()
        if args.stage1:
            pipeline.model_cfgs["denoiser"] = args.stage1
        if args.stage2:
            pipeline.model_cfgs["slat_denoiser"] = args.stage2
        if not args.low_vram:
            outputs = pipeline.run(
                paths,
                seed=args.seed,
                steps=args.steps,
                remove_background=args.remove_background,
            )
            emi.export_result(
                outputs,
                outputs["extrinsics"],
                outputs["intrinsics"],
                outputs["inputs"],
                output_dir,
            )
            return
        model_cfgs = pipeline.model_cfgs
        options = dict(pipeline.inference)
        from trellis.pipelines.gensfm import NATIVE_INFERENCE_OPTIONS
        options.update(NATIVE_INFERENCE_OPTIONS)
        for key in ("ssuv_sampler_args", "slat_sampler_args"):
            if args.steps is not None:
                options[key]["params"]["steps"] = args.steps
        crop_ratio = pipeline.preprocessing["crop_size_ratio"]

    print("[mrGenSfM Low-VRAM] Initializing sequential CPU<->GPU stage offloading...", flush=True)
    inputs = InputViews(paths, crop_ratio, args.remove_background)
    runtime_dir = output_dir / "_runtime_lowvram"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    evaluator = evaluators.VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator(
        model_cfgs, inputs, output_dir=str(runtime_dir), **options
    )
    for name, model in evaluator.models.items():
        model.eval()
        model.cpu()
    torch.cuda.empty_cache()

    # Step 1: DINOv2 image conditioning on GPU, then offload DINOv2 to CPU
    evaluator._init_image_cond_model()
    with torch.no_grad():
        cond = inputs.condition[None].to(evaluator.device)
        # Stage 1 models to GPU
        for role in ("denoiser", "ss_decoder", "suv_decoder"):
            if role in evaluator.models:
                evaluator.models[role].to(evaluator.device)
        print("[mrGenSfM Low-VRAM] Running Stage 1 (pose & coarse shape)...", flush=True)
        ssuv, _ = evaluator._sample_ssuv(cond=cond)
        extrinsics, intrinsics, _ = evaluator._solve_pose_from_ssuv(
            ssuv["ssuv_recon"].flatten(0, 1), ssuv["uv_volume_pred"].flatten(0, 1)
        )

        # Offload Stage 1 models to CPU RAM before allocating Stage 2 tensors
        for role in ("denoiser", "ss_decoder", "suv_decoder"):
            if role in evaluator.models:
                evaluator.models[role].cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Move Stage 2 models to GPU (needed by _make_randn_sparse_latent_from_ss for slat_denoiser.device)
        for role in ("slat_denoiser", "slat_decoder_gs", "slat_decoder_mesh"):
            if role in evaluator.models:
                evaluator.models[role].to(evaluator.device)
        noise = evaluator._make_randn_sparse_latent_from_ss(ssuv["ss_recon"])
        uvs = evaluator._make_sparse_uvs_from_pose(noise, extrinsics[None], intrinsics[None])
        print("[mrGenSfM Low-VRAM] Running Stage 2 (appearance & fine geometry)...", flush=True)
        outputs = evaluator._sample_slat(cond=cond, noise=noise, uvs=uvs)

        if getattr(evaluator, "image_cond_model", None) and "model" in evaluator.image_cond_model:
            evaluator.image_cond_model["model"].cpu()
        for role in ("slat_denoiser", "slat_decoder_gs", "slat_decoder_mesh"):
            if role in evaluator.models:
                evaluator.models[role].cpu()
        gc.collect()
        torch.cuda.empty_cache()

    emi.export_result(outputs, extrinsics, intrinsics, inputs, output_dir)
    run_meta = {
        "seed": args.seed,
        "num_views": len(paths),
        "low_vram": True,
        "pretrained": args.pretrained,
    }
    (output_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
'''


def windows_to_wsl_path(win_path: str | Path) -> str:
    """
    Convert a Windows filesystem path (e.g., 'D:\\Meshroom\\mrGenSfM') to a WSL2 mount path
    ('/mnt/d/Meshroom/mrGenSfM'). Passes POSIX paths through unchanged.
    """
    raw = str(win_path).strip()
    if not raw:
        raise ValueError("Path cannot be empty")
    if raw.startswith("/"):
        return raw

    normalized = raw.replace("\\", "/")
    match = re.match(r"^([a-zA-Z]):/(.*)$", normalized)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)
        return f"/mnt/{drive}/{rest}".rstrip("/") if rest else f"/mnt/{drive}"
    return normalized


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert a WSL2 mount path ('/mnt/d/Meshroom/...') back to a Windows path ('D:\\Meshroom\\...')."""
    cleaned = wsl_path.strip()
    match = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", cleaned)
    if match:
        drive = match.group(1).upper()
        rest = (match.group(2) or "").replace("/", "\\")
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    return cleaned


def resolve_gensfm_repo_path() -> Path:
    """Locate the local clone of dotb1ue/GenSfm on disk."""
    env_repo = os.environ.get("GENSFM_REPO_PATH", "").strip()
    candidates = [
        Path(env_repo) if env_repo else None,
        Path("D:/Meshroom/GenSfm"),
        Path(__file__).resolve().parents[2] / "GenSfm",
    ]
    for cand in candidates:
        if cand is not None and cand.is_dir() and (cand / "example_multi_image.py").is_file():
            return cand.resolve()
    return Path("D:/Meshroom/GenSfm")


def run_gensfm_via_wsl(
    image_paths: Sequence[Path],
    output_dir: Path,
    pretrained_model: str = "dotblue/GenSfm",
    stage1_checkpoint: str = "",
    stage2_checkpoint: str = "",
    seed: int = 1,
    steps: int = 25,
    low_vram_mode: bool = True,
    remove_background: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Execute native GenSfM inside WSL2 Ubuntu using the Low-VRAM wrapper script.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gensfm_repo = resolve_gensfm_repo_path()
    if not (gensfm_repo / "example_multi_image.py").is_file():
        raise FileNotFoundError(
            f"GenSfM repository not found at {gensfm_repo}. Set GENSFM_REPO_PATH in .env."
        )

    wrapper_file = output_dir / "_mr_gensfm_low_vram_exec.py"
    wrapper_file.write_text(LOW_VRAM_WRAPPER_SCRIPT, encoding="utf-8")

    wsl_distro = os.environ.get("GENSFM_WSL_DISTRO", "Ubuntu").strip() or "Ubuntu"
    wsl_conda_env = os.environ.get("GENSFM_WSL_CONDA_ENV", "gensfm").strip() or "gensfm"

    wsl_repo = windows_to_wsl_path(gensfm_repo)
    wsl_out = windows_to_wsl_path(output_dir)
    wsl_wrapper = windows_to_wsl_path(wrapper_file)
    wsl_imgs = " ".join(shlex.quote(windows_to_wsl_path(p.resolve())) for p in image_paths)

    flags = f"--seed {int(seed)} --steps {int(steps)}"
    if stage1_checkpoint.strip():
        flags += f" --stage1 {shlex.quote(windows_to_wsl_path(stage1_checkpoint.strip()))}"
    if stage2_checkpoint.strip():
        flags += f" --stage2 {shlex.quote(windows_to_wsl_path(stage2_checkpoint.strip()))}"
    if low_vram_mode:
        flags += " --low-vram"
    if remove_background:
        flags += " --remove-background"

    bash_cmd = (
        "export ATTN_BACKEND=sdpa ATTN_BACKEND_WITH_ATTN_BIAS=sdpa SPCONV_ALGO=native "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
        "for cpath in \"$HOME/miniforge3\" \"$HOME/mambaforge\" \"$HOME/miniconda3\" \"$HOME/anaconda3\" \"/opt/conda\"; do "
        "  if [ -f \"$cpath/etc/profile.d/conda.sh\" ]; then . \"$cpath/etc/profile.d/conda.sh\"; conda activate "
        f"{shlex.quote(wsl_conda_env)} 2>/dev/null || true; break; fi; "
        "done; "
        f"cd {shlex.quote(wsl_repo)} && "
        f"python3 {shlex.quote(wsl_wrapper)} "
        f"--gensfm-root {shlex.quote(wsl_repo)} "
        f"--images {wsl_imgs} "
        f"--output-dir {shlex.quote(wsl_out)} "
        f"--pretrained {shlex.quote(pretrained_model)} "
        f"{flags}"
    )

    cmd = ["wsl.exe", "-d", wsl_distro, "--", "bash", "-lc", bash_cmd]
    if logger:
        logger.info(f"Executing GenSfM in WSL2 ({wsl_distro}): {bash_cmd}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"WSL2 GenSfM execution exited with code {proc.returncode}.\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    return output_dir


def run_gensfm_via_docker(
    image_paths: Sequence[Path],
    output_dir: Path,
    pretrained_model: str = "dotblue/GenSfm",
    stage1_checkpoint: str = "",
    stage2_checkpoint: str = "",
    seed: int = 1,
    steps: int = 25,
    low_vram_mode: bool = True,
    remove_background: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Execute native GenSfM inside a GPU-enabled Docker container (following the mrGSplat pattern).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gensfm_repo = resolve_gensfm_repo_path()

    wrapper_file = output_dir / "_mr_gensfm_low_vram_exec.py"
    wrapper_file.write_text(LOW_VRAM_WRAPPER_SCRIPT, encoding="utf-8")

    docker_image = os.environ.get("GENSFM_DOCKER_IMAGE", "gensfm:cu128").strip() or "gensfm:cu128"
    container_imgs: List[str] = [f"/workspace/output/{p.name}" for p in image_paths]

    cmd: List[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "ATTN_BACKEND=sdpa",
        "-e",
        "ATTN_BACKEND_WITH_ATTN_BIAS=sdpa",
        "-e",
        "SPCONV_ALGO=native",
        "-e",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-v",
        f"{output_dir}:/workspace/output",
        "-v",
        f"{gensfm_repo}:/workspace/GenSfm",
        docker_image,
        "python3",
        "/workspace/output/_mr_gensfm_low_vram_exec.py",
        "--gensfm-root",
        "/workspace/GenSfm",
        "--images",
        *container_imgs,
        "--output-dir",
        "/workspace/output",
        "--pretrained",
        pretrained_model,
        "--seed",
        str(int(seed)),
        "--steps",
        str(int(steps)),
    ]
    if stage1_checkpoint.strip():
        cmd.extend(["--stage1", stage1_checkpoint.strip()])
    if stage2_checkpoint.strip():
        cmd.extend(["--stage2", stage2_checkpoint.strip()])
    if low_vram_mode:
        cmd.append("--low-vram")
    if remove_background:
        cmd.append("--remove-background")

    if logger:
        logger.info(f"Executing GenSfM via Docker ({docker_image}): {' '.join(cmd)}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Docker GenSfM execution exited with code {proc.returncode}.\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    return output_dir


def run_gensfm_via_local_venv(
    image_paths: Sequence[Path],
    output_dir: Path,
    pretrained_model: str = "dotblue/GenSfm",
    stage1_checkpoint: str = "",
    stage2_checkpoint: str = "",
    seed: int = 1,
    steps: int = 25,
    low_vram_mode: bool = True,
    remove_background: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Execute native GenSfM directly in a local Python environment.
    """
    import sys

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gensfm_repo = resolve_gensfm_repo_path()

    wrapper_file = output_dir / "_mr_gensfm_low_vram_exec.py"
    wrapper_file.write_text(LOW_VRAM_WRAPPER_SCRIPT, encoding="utf-8")

    python_exe = os.environ.get("GENSFM_PYTHON_EXE", sys.executable)
    cmd: List[str] = [
        python_exe,
        str(wrapper_file),
        "--gensfm-root",
        str(gensfm_repo),
        "--images",
        *[str(p.resolve()) for p in image_paths],
        "--output-dir",
        str(output_dir),
        "--pretrained",
        pretrained_model,
        "--seed",
        str(int(seed)),
        "--steps",
        str(int(steps)),
    ]
    if stage1_checkpoint.strip():
        cmd.extend(["--stage1", stage1_checkpoint.strip()])
    if stage2_checkpoint.strip():
        cmd.extend(["--stage2", stage2_checkpoint.strip()])
    if low_vram_mode:
        cmd.append("--low-vram")
    if remove_background:
        cmd.append("--remove-background")

    if logger:
        logger.info(f"Executing GenSfM via local Python ({python_exe})")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Local venv GenSfM execution exited with code {proc.returncode}.\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )

    return output_dir
