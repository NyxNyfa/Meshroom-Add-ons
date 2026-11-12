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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


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
    parser.add_argument("--fast-texture", action="store_true")
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
    import trellis.evaluators.base as eval_base
    import trellis.representations.mesh.cube2mesh as c2m
    import trellis.utils.postprocessing_utils as pu

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
        all_cfgs, options, crop_ratio = emi.inference_config(
            args.stage1, args.stage2, len(paths), args.steps if args.steps is not None else 25
        )
    else:
        pipeline = GenSfMPipeline.from_pretrained(args.pretrained)
        if not args.low_vram:
            pipeline = pipeline.cuda()
            if args.stage1:
                pipeline.model_cfgs["denoiser"] = args.stage1
            if args.stage2:
                pipeline.model_cfgs["slat_denoiser"] = args.stage2
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

        all_cfgs = dict(pipeline.model_cfgs)
        if args.stage1:
            all_cfgs["denoiser"] = args.stage1
        if args.stage2:
            all_cfgs["slat_denoiser"] = args.stage2
        options = dict(pipeline.inference)
        from trellis.pipelines.gensfm import NATIVE_INFERENCE_OPTIONS
        options.update(NATIVE_INFERENCE_OPTIONS)
        for key in ("ssuv_sampler_args", "slat_sampler_args"):
            if args.steps is not None:
                options[key]["params"]["steps"] = args.steps
        crop_ratio = pipeline.preprocessing["crop_size_ratio"]
        del pipeline
        gc.collect()

    if args.fast_texture:
        orig_to_glb = pu.to_glb
        def fast_to_glb(*g_args, **g_kwargs):
            orig_bake = pu.bake_texture
            pu.bake_texture = lambda *b_args, **b_kwargs: orig_bake(*b_args, **dict(b_kwargs, mode="fast"))
            try:
                return orig_to_glb(*g_args, **g_kwargs)
            finally:
                pu.bake_texture = orig_bake
        pu.to_glb = fast_to_glb

    print("[mrGenSfM Low-VRAM] Initializing sequential Stage 1 execution...", flush=True)
    inputs = InputViews(paths, crop_ratio, args.remove_background)
    runtime_dir = output_dir / "_runtime_lowvram"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Evaluator ONLY with Stage 1 models (denoiser, ss_decoder, suv_decoder)
    stage1_cfgs = {k: all_cfgs[k] for k in ("denoiser", "ss_decoder", "suv_decoder") if k in all_cfgs}
    evaluator = evaluators.VisualImageConditionedMultiPoseSparseFlowMatchingEvaluator(
        stage1_cfgs, inputs, output_dir=str(runtime_dir), **options
    )
    for model in evaluator.models.values():
        model.eval()
    evaluator._init_image_cond_model()

    with torch.no_grad():
        cond = inputs.condition[None].to(evaluator.device)
        print("[mrGenSfM Low-VRAM] Running Stage 1 (sparse structure & camera poses)...", flush=True)
        ssuv, _ = evaluator._sample_ssuv(cond=cond)
        extrinsics, intrinsics, _ = evaluator._solve_pose_from_ssuv(
            ssuv["ssuv_recon"].flatten(0, 1), ssuv["uv_volume_pred"].flatten(0, 1)
        )
        print(f"[mrGenSfM Low-VRAM] Solved camera poses for {len(paths)} views.", flush=True)

        # Pre-compute Stage 2 visual conditioning with DINOv2
        print("[mrGenSfM Low-VRAM] Computing visual latent conditioning...", flush=True)
        slat_cond_args = evaluator.get_inference_cond(cond, type="slat")

        # Stage 1 and DINOv2 are completely finished; free them to release ~2.5GB RAM & VRAM
        for role in ("denoiser", "ss_decoder", "suv_decoder"):
            if role in evaluator.models:
                del evaluator.models[role]
        if hasattr(evaluator, "image_cond_model"):
            del evaluator.image_cond_model
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Lazy-load Stage 2 slat_denoiser on demand
        print("[mrGenSfM Low-VRAM] Loading Stage 2 (SLat appearance flow model)...", flush=True)
        evaluator.models["slat_denoiser"] = eval_base.build_model(all_cfgs["slat_denoiser"], device="cuda")
        evaluator.load({"slat_denoiser": all_cfgs["slat_denoiser"]})
        evaluator.models["slat_denoiser"].eval()

        noise = evaluator._make_randn_sparse_latent_from_ss(ssuv["ss_recon"])
        slat_cond_args["uvs"] = evaluator._make_sparse_uvs_from_pose(noise, extrinsics[None], intrinsics[None])

        print("[mrGenSfM Low-VRAM] Running Stage 2 (SLat diffusion sampling)...", flush=True)
        res = evaluator.slat_sampler.sample(
            model=evaluator.models["slat_denoiser"],
            noise=noise,
            **slat_cond_args,
            **evaluator.slat_sampler_params,
            verbose=False,
        )
        pred_x_0 = res.samples

        # Free Stage 2 SLat denoiser (~2.3GB) before running mesh & gaussian decoders
        del evaluator.models["slat_denoiser"]
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Decode 3D Mesh with CPU-offloaded FlexiCubes extraction (prevents 2.5GB dense grid VRAM spike)
        print("[mrGenSfM Low-VRAM] Decoding 3D Mesh geometry with CPU FlexiCubes...", flush=True)
        evaluator.models["slat_decoder_mesh"] = eval_base.build_model(all_cfgs["slat_decoder_mesh"], device="cuda")
        evaluator.load({"slat_decoder_mesh": all_cfgs["slat_decoder_mesh"]})
        decoder_mesh = evaluator.models["slat_decoder_mesh"]
        decoder_mesh.eval()

        decoder_mesh.mesh_extractor = c2m.SparseFeatures2Mesh(
            device="cpu",
            res=decoder_mesh.resolution * 4,
            use_color=decoder_mesh.rep_config.get("use_color", False),
        )
        def cpu_to_representation(x):
            return [decoder_mesh.mesh_extractor(x[i].cpu(), training=decoder_mesh.training) for i in range(x.shape[0])]
        decoder_mesh.to_representation = cpu_to_representation

        mesh_output = sum([decoder_mesh(pred_x_0[i:i+1]) for i in range(pred_x_0.shape[0])], [])
        del evaluator.models["slat_decoder_mesh"]
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Decode 3D Gaussians
        print("[mrGenSfM Low-VRAM] Decoding 3D Gaussian Splats...", flush=True)
        evaluator.models["slat_decoder_gs"] = eval_base.build_model(all_cfgs["slat_decoder_gs"], device="cuda")
        evaluator.load({"slat_decoder_gs": all_cfgs["slat_decoder_gs"]})
        evaluator.models["slat_decoder_gs"].eval()
        gs_output = sum([evaluator.models["slat_decoder_gs"](pred_x_0[i:i+1]) for i in range(pred_x_0.shape[0])], [])
        del evaluator.models["slat_decoder_gs"]
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()

    outputs = {"mesh": mesh_output, "gaussian": gs_output}
    print("[mrGenSfM Low-VRAM] Exporting GLB, PLY, and camera reprojections...", flush=True)
    emi.export_result(outputs, extrinsics, intrinsics, inputs, output_dir)

    run_meta = {
        "seed": args.seed,
        "num_views": len(paths),
        "low_vram": True,
        "pretrained": args.pretrained,
    }
    (output_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\\n", encoding="utf-8")
    print(f"[mrGenSfM Low-VRAM] Reconstruction finished successfully: {output_dir}", flush=True)


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
    optimize_texture: bool = True,
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
    wrapper_file.write_bytes(LOW_VRAM_WRAPPER_SCRIPT.replace("\r\n", "\n").encode("utf-8"))

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
    if not optimize_texture:
        flags += " --fast-texture"
    if remove_background:
        flags += " --remove-background"

    runner_file = output_dir / "_mr_gensfm_runner.sh"
    runner_script = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "\n"
        "# Source Conda if present\n"
        "for cpath in \"$HOME/miniforge3\" \"$HOME/mambaforge\" \"$HOME/miniconda3\" \"$HOME/anaconda3\" \"/opt/conda\"; do\n"
        "  if [ -f \"$cpath/etc/profile.d/conda.sh\" ]; then\n"
        "    . \"$cpath/etc/profile.d/conda.sh\"\n"
        f"    conda activate {shlex.quote(wsl_conda_env)} 2>/dev/null || true\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "\n"
        "# Resolve explicit Conda Python executable\n"
        "PY_BIN=\"\"\n"
        f"for candidate in \\\n"
        f"  \"$HOME/miniforge3/envs/{wsl_conda_env}/bin/python\" \\\n"
        f"  \"$HOME/miniforge3/envs/{wsl_conda_env}/bin/python3\" \\\n"
        f"  \"$HOME/miniconda3/envs/{wsl_conda_env}/bin/python\" \\\n"
        f"  \"$HOME/miniconda3/envs/{wsl_conda_env}/bin/python3\" \\\n"
        f"  \"$HOME/mambaforge/envs/{wsl_conda_env}/bin/python\" \\\n"
        f"  \"$HOME/mambaforge/envs/{wsl_conda_env}/bin/python3\" \\\n"
        f"  \"/opt/conda/envs/{wsl_conda_env}/bin/python\" \\\n"
        f"  \"/opt/conda/envs/{wsl_conda_env}/bin/python3\"; do\n"
        "  if [ -x \"$candidate\" ]; then\n"
        "    PY_BIN=\"$candidate\"\n"
        "    break\n"
        "  fi\n"
        "done\n"
        "\n"
        "if [ -z \"$PY_BIN\" ]; then\n"
        "  if command -v python3 >/dev/null 2>&1; then\n"
        "    PY_BIN=\"$(which python3)\"\n"
        "  else\n"
        "    PY_BIN=\"python\"\n"
        "  fi\n"
        "fi\n"
        "\n"
        "export ATTN_BACKEND=sdpa\n"
        "export ATTN_BACKEND_WITH_ATTN_BIAS=sdpa\n"
        "export SPCONV_ALGO=native\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "export PYTORCH_ALLOC_CONF=expandable_segments:True\n"
        "\n"
        f"cd {shlex.quote(wsl_repo)}\n"
        f"exec \"$PY_BIN\" {shlex.quote(wsl_wrapper)} \\\n"
        f"  --gensfm-root {shlex.quote(wsl_repo)} \\\n"
        f"  --images {wsl_imgs} \\\n"
        f"  --output-dir {shlex.quote(wsl_out)} \\\n"
        f"  --pretrained {shlex.quote(pretrained_model)} \\\n"
        f"  {flags}\n"
    )
    runner_file.write_bytes(runner_script.replace("\r\n", "\n").encode("utf-8"))

    wsl_runner = windows_to_wsl_path(runner_file)
    cmd = ["wsl.exe", "-d", wsl_distro, "--", "bash", wsl_runner]
    if logger:
        logger.info(f"Executing GenSfM in WSL2 ({wsl_distro}) via runner script: {wsl_runner}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            line_str = line.rstrip()
            if line_str:
                captured_lines.append(line_str)
                if logger:
                    logger.info(f"[WSL2 GenSfM] {line_str}")
        proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        combined_output = "\n".join(captured_lines)
        raise RuntimeError(
            f"WSL2 GenSfM execution exited with code {proc.returncode}.\n"
            f"Output:\n{combined_output}"
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
    optimize_texture: bool = True,
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
    if not optimize_texture:
        cmd.append("--fast-texture")
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
    optimize_texture: bool = True,
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
    if not optimize_texture:
        cmd.append("--fast-texture")
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
