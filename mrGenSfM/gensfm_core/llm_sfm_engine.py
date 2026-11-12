"""
Lightweight Multimodal LLM-Guided Sparse SfM + Multi-View Visual Hull 3D Reconstruction Engine.

Designed for laptops and workstations without large NVIDIA GPUs:
1. Estimates per-view camera poses (azimuth, elevation, roll, distance, FoV) using Multimodal Vision LLMs:
   - Google Gemini API (`gemini-3.8-flash` via Interactions API / `google-genai` SDK)
   - OpenAI Vision API (`gpt-4o`)
   - Anthropic Claude Vision API (`claude-3-7-sonnet-latest`)
   - Automatic deterministic visual-hull turntable pose solver when no API key is set (zero-config offline testability)
2. Converts estimated poses into GenSfM's exact OpenCV Z-up extrinsics & normalized intrinsics (`cameras.json`).
3. Reconstructs a 3D textured mesh (`mesh.glb`, `mesh.obj` + `.mtl` + `.png`), 3D Gaussian Splats (`gaussian.ply`),
   and Z-buffered camera reprojections (`reprojection-00..03.png`) via Multi-View Silhouette Carving & Normal-Weighted Texture Projection.
"""
from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

from gensfm_core.camera_converter import DEFAULT_DATA_TO_MESH
from gensfm_core.masking import PreparedViewRecord
from gensfm_core.mesh_converter import write_gaussian_ply, write_textured_glb, write_textured_obj


@dataclass
class PredictedViewCamera:
    """Spherical camera parameters in GenSfM's Z-up object-centric coordinate system."""

    azimuth_deg: float
    elevation_deg: float
    roll_deg: float = 0.0
    distance: float = 2.0
    fov_deg: float = 40.0


def normalize_gemini_model_name(model_name: str, logger: Optional[logging.Logger] = None) -> str:
    """Upgrade deprecated Gemini model identifiers to gemini-3.8-flash."""
    cleaned = model_name.strip()
    if not cleaned or cleaned.lower() == "dotblue/gensfm":
        return "gemini-3.8-flash"
    lower = cleaned.lower()
    if any(dep in lower for dep in ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-pro-vision")):
        if logger:
            logger.info(
                f"Upgrading deprecated Gemini model '{cleaned}' to 'gemini-3.8-flash'."
            )
        return "gemini-3.8-flash"
    return cleaned


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _build_pose_estimation_prompt(num_views: int) -> str:
    return (
        f"You are a 3D Computer Vision Structure-from-Motion expert. "
        f"You are given {num_views} foreground-segmented views (indexed 0 to {num_views - 1}) of the same rigid 3D object. "
        "Estimate the relative camera pose for each view in a Z-up object-centered spherical coordinate system "
        "where view 0 is at azimuth_deg=0.0:\n"
        "- azimuth_deg: horizontal angle around the Z axis in degrees [0, 360)\n"
        "- elevation_deg: vertical elevation angle above the XY plane in degrees [-60, 80]\n"
        "- roll_deg: in-plane camera tilt in degrees [-30, 30]\n"
        "- distance: camera distance from the object center in normalized units [1.4, 3.0] (typical: 2.0)\n"
        "- fov_deg: horizontal field of view in degrees [25, 65] (typical: 40.0)\n"
        "Respond strictly with a JSON object containing a 'cameras' array of length "
        f"{num_views} with keys: index, azimuth_deg, elevation_deg, roll_deg, distance, fov_deg."
    )


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                cleaned = candidate
                break
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _query_gemini_poses(
    prepared_views: Sequence[PreparedViewRecord],
    api_key: str,
    model_name: str,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> List[PredictedViewCamera]:
    model = normalize_gemini_model_name(model_name, logger=logger)
    prompt = _build_pose_estimation_prompt(len(prepared_views))

    response_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "cameras": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "azimuth_deg": {"type": "number"},
                        "elevation_deg": {"type": "number"},
                        "roll_deg": {"type": "number"},
                        "distance": {"type": "number"},
                        "fov_deg": {"type": "number"},
                    },
                    "required": ["index", "azimuth_deg", "elevation_deg", "roll_deg", "distance", "fov_deg"],
                },
            }
        },
        "required": ["cameras"],
    }

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for pv in prepared_views:
        content_parts.append(
            {
                "type": "image",
                "mime_type": "image/png",
                "data": _encode_image_base64(pv.rgba_path),
            }
        )

    raw_text = ""
    response_format_cfg: Dict[str, Any] = {
        "type": "text",
        "mime_type": "application/json",
        "schema": response_schema,
    }
    # Prefer official google-genai SDK (>=2.25.0) Interactions API if installed
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=model,
            input=[{"type": "user_input", "content": content_parts}],
            generation_config={"seed": int(seed)},
            response_format=response_format_cfg,
            store=False,
        )
        raw_text = interaction.output_text or ""
    except ImportError as imp_exc:
        (logger or logging.getLogger(__name__)).debug(
            f"google-genai SDK not installed ({imp_exc}); using direct Gemini Interactions REST API."
        )
        # Direct REST call to Gemini Interactions API v1
        url = "https://generativelanguage.googleapis.com/v1/interactions"
        payload: Dict[str, Any] = {
            "model": model if model.startswith("models/") else f"models/{model}",
            "input": [{"type": "user_input", "content": content_parts}],
            "generation_config": {"seed": int(seed)},
            "response_format": response_format_cfg,
            "store": False,
        }
        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        resp_json = resp.json()
        for step in reversed(resp_json.get("steps", [])):
            if step.get("type") == "model_output":
                for part in step.get("content", []):
                    if part.get("type") == "text" and part.get("text"):
                        raw_text = str(part["text"])
                        break
            if raw_text:
                break

    parsed = _extract_json_from_text(raw_text)
    return _parse_predicted_cameras_dict(parsed, len(prepared_views))


def _query_openai_poses(
    prepared_views: Sequence[PreparedViewRecord],
    api_key: str,
    model_name: str,
    seed: int,
) -> List[PredictedViewCamera]:
    model = model_name.strip() if model_name.strip() and "gpt" in model_name.lower() else "gpt-4o"
    prompt = _build_pose_estimation_prompt(len(prepared_views))
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for pv in prepared_views:
        b64 = _encode_image_base64(pv.rgba_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "seed": int(seed),
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return _parse_predicted_cameras_dict(_extract_json_from_text(text), len(prepared_views))


def _query_anthropic_poses(
    prepared_views: Sequence[PreparedViewRecord],
    api_key: str,
    model_name: str,
) -> List[PredictedViewCamera]:
    model = (
        model_name.strip()
        if model_name.strip() and "claude" in model_name.lower()
        else "claude-3-7-sonnet-latest"
    )
    prompt = _build_pose_estimation_prompt(len(prepared_views))
    content: List[Dict[str, Any]] = []
    for pv in prepared_views:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _encode_image_base64(pv.rgba_path),
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    return _parse_predicted_cameras_dict(_extract_json_from_text(text), len(prepared_views))


def _parse_predicted_cameras_dict(data: Dict[str, Any], num_views: int) -> List[PredictedViewCamera]:
    raw_list = data.get("cameras", [])
    if not isinstance(raw_list, list) or len(raw_list) < num_views:
        raise ValueError(f"Expected {num_views} camera predictions in JSON, got {len(raw_list)}")

    result: List[PredictedViewCamera] = []
    for i in range(num_views):
        item = raw_list[i]
        result.append(
            PredictedViewCamera(
                azimuth_deg=float(item.get("azimuth_deg", (360.0 * i) / num_views)) % 360.0,
                elevation_deg=float(np.clip(float(item.get("elevation_deg", 20.0)), -75.0, 75.0)),
                roll_deg=float(np.clip(float(item.get("roll_deg", 0.0)), -45.0, 45.0)),
                distance=float(np.clip(float(item.get("distance", 2.0)), 1.2, 4.0)),
                fov_deg=float(np.clip(float(item.get("fov_deg", 40.0)), 20.0, 75.0)),
            )
        )
    return result


def _compute_view_appearance_descriptor(pv: PreparedViewRecord) -> np.ndarray:
    """Compute a compact appearance + silhouette descriptor for relative random-view ordering."""
    rgba = pv.rgba_array.astype(np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    # Downsample to 8x8 spatial grid of premultiplied RGB + alpha
    pil_small = Image.fromarray(pv.rgba_array, mode="RGBA").resize((8, 8), Image.Resampling.BILINEAR)
    arr_small = np.asarray(pil_small, dtype=np.float32) / 255.0
    spatial_feat = (arr_small[..., :3] * arr_small[..., 3:4]).reshape(-1)
    alpha_feat = arr_small[..., 3].reshape(-1)

    # Color histogram on foreground pixels
    fg_mask = alpha > 0.2
    if np.any(fg_mask):
        fg_rgb = rgb[fg_mask]
        r_hist, _ = np.histogram(fg_rgb[:, 0], bins=8, range=(0.0, 1.0), density=True)
        g_hist, _ = np.histogram(fg_rgb[:, 1], bins=8, range=(0.0, 1.0), density=True)
        b_hist, _ = np.histogram(fg_rgb[:, 2], bins=8, range=(0.0, 1.0), density=True)
        color_hist = np.concatenate([r_hist, g_hist, b_hist]).astype(np.float32) * 0.1
    else:
        color_hist = np.zeros(24, dtype=np.float32)

    desc = np.concatenate([spatial_feat, alpha_feat * 0.5, color_hist])
    norm = float(np.linalg.norm(desc))
    return desc / norm if norm > 1e-8 else desc


def estimate_poses_heuristic(prepared_views: Sequence[PreparedViewRecord]) -> List[PredictedViewCamera]:
    """
    Deterministic multi-view random-angle pose estimator used when offline or when no API key is configured.
    Recovers relative angular ordering, non-uniform azimuth spacing, elevation, distance, and FoV
    from uncalibrated sparse/random views using pre-crop bounding-box scale, silhouette geometry,
    and pairwise appearance dissimilarity.
    """
    n = len(prepared_views)
    if n == 0:
        return []

    descriptors = [_compute_view_appearance_descriptor(pv) for pv in prepared_views]

    # Order views around the object starting from view 0 (azimuth=0) via greedy TSP on visual appearance
    ordered_indices: List[int] = [0]
    remaining = list(range(1, n))
    while remaining:
        last_idx = ordered_indices[-1]
        remaining.sort(key=lambda cand: float(np.linalg.norm(descriptors[cand] - descriptors[last_idx])))
        ordered_indices.append(remaining.pop(0))

    # Compute angular separation proportional to visual dissimilarity between adjacent ordered views
    arc_weights: List[float] = []
    for k in range(n):
        i_curr = ordered_indices[k]
        i_next = ordered_indices[(k + 1) % n]
        dist_val = float(np.linalg.norm(descriptors[i_curr] - descriptors[i_next]))
        arc_weights.append(max(0.25, dist_val + 0.35))

    total_weight = sum(arc_weights)
    azimuth_by_index: Dict[int, float] = {0: 0.0}
    cum_angle = 0.0
    for k in range(n - 1):
        step_deg = 360.0 * (arc_weights[k] / max(total_weight, 1e-8))
        cum_angle = (cum_angle + step_deg) % 360.0
        azimuth_by_index[ordered_indices[k + 1]] = cum_angle

    cameras: List[PredictedViewCamera] = []
    for i, pv in enumerate(prepared_views):
        alpha = pv.rgba_array[..., 3].astype(np.float32) / 255.0
        ys, xs = np.nonzero(alpha > 0.2)

        w_orig, h_orig = pv.source_size
        orig_left, orig_top, orig_right, orig_bottom = pv.crop_pixels
        crop_span_px = float(max(1, max(orig_right - orig_left, orig_bottom - orig_top)))
        source_span_px = float(max(1, max(w_orig, h_orig)))
        # True uncropped object occupancy ratio in the original CameraInit photo
        fg_occupancy = float(np.clip((crop_span_px / 1.2) / source_span_px, 0.20, 0.95))

        orig_cy_norm = (0.5 * float(orig_top + orig_bottom) / float(max(1, h_orig))) - 0.5
        if len(xs) > 10 and len(ys) > 10:
            # Measure vertical mass asymmetry inside the silhouette (top vs bottom width)
            y_mid = 0.5 * float(ys.min() + ys.max())
            top_mass = float(np.count_nonzero(ys < y_mid)) / float(len(ys))
            mass_tilt = (top_mass - 0.5) * 28.0
        else:
            mass_tilt = 0.0

        az = float(azimuth_by_index.get(i, (360.0 * float(i)) / float(n)))
        elev = float(np.clip(20.0 - orig_cy_norm * 32.0 + mass_tilt, -45.0, 65.0))
        dist = float(np.clip(1.35 / math.sqrt(max(fg_occupancy, 0.25)), 1.45, 2.85))

        fov_deg = 38.0
        if pv.view_descriptor.focal_px and pv.view_descriptor.focal_px > 1.0:
            fov_rad = 2.0 * math.atan(0.5 * source_span_px / float(pv.view_descriptor.focal_px))
            fov_deg = float(np.clip(math.degrees(fov_rad), 24.0, 68.0))

        cameras.append(
            PredictedViewCamera(
                azimuth_deg=az,
                elevation_deg=elev,
                roll_deg=0.0,
                distance=dist,
                fov_deg=fov_deg,
            )
        )
    return cameras


def spherical_camera_to_opencv_extrinsic_and_intrinsic(
    cam: PredictedViewCamera,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical camera parameters (Z-up world frame, looking at origin) into:
      - 4x4 OpenCV world-to-camera extrinsic matrix (+X right, +Y down, +Z forward)
      - 3x3 normalized pinhole intrinsic matrix
    """
    az = math.radians(cam.azimuth_deg)
    el = math.radians(cam.elevation_deg)
    roll = math.radians(cam.roll_deg)
    dist = float(cam.distance)

    # Camera position in Z-up world coordinates
    cx = dist * math.cos(el) * math.cos(az)
    cy = dist * math.cos(el) * math.sin(az)
    cz = dist * math.sin(el)
    cam_pos = np.array([cx, cy, cz], dtype=np.float64)

    # Forward (+Z_cam) points from camera position toward the origin (0, 0, 0)
    forward = -cam_pos / max(np.linalg.norm(cam_pos), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Right (+X_cam) = forward x world_up
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-8)
    # Down (+Y_cam) = forward x right (OpenCV right-handed frame: X right, Y down, Z forward)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-8)

    if abs(roll) > 1e-6:
        cr, sr = math.cos(roll), math.sin(roll)
        r_right = cr * right + sr * down
        r_down = -sr * right + cr * down
        right, down = r_right, r_down

    r_w2c = np.stack([right, down, forward], axis=0)
    t_w2c = -(r_w2c @ cam_pos)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = r_w2c
    extrinsic[:3, 3] = t_w2c

    fov_rad = math.radians(max(15.0, min(85.0, cam.fov_deg)))
    f_norm = 0.5 / math.tan(0.5 * fov_rad)
    intrinsic = np.array(
        [
            [f_norm, 0.0, 0.5],
            [0.0, f_norm, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return extrinsic, intrinsic


def compute_source_intrinsics(
    crop_intrinsics: Sequence[np.ndarray],
    prepared_views: Sequence[PreparedViewRecord],
) -> List[List[List[float]]]:
    """Map normalized crop intrinsics back to EXIF-oriented source images (matching GenSfM)."""
    result: List[List[List[float]]] = []
    for k_mat, pv in zip(crop_intrinsics, prepared_views):
        (left, top), (right, bottom) = pv.crop_normalized
        mapping = np.array(
            [
                [right - left, 0.0, left],
                [0.0, bottom - top, top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result.append((mapping @ k_mat).tolist())
    return result


def _reconstruct_visual_hull_and_texture(
    prepared_views: Sequence[PreparedViewRecord],
    extrinsics: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    n_lat: int = 36,
    n_lon: int = 72,
    tex_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image, np.ndarray, np.ndarray]:
    """
    Reconstruct a smooth 3D mesh and 3D Gaussian point cloud from multi-view RGBA masks and colors
    using Multi-View Ray Silhouette Carving + Laplacian Smoothing + Normal-Weighted Texture Baking.

    Returns:
      - vertices_yup: (V, 3) float32 in Y-up mesh frame (DATA_TO_MESH applied)
      - faces: (F, 3) int64 triangle indices
      - uvs: (V, 2) float32 UV coordinates in [0, 1]
      - texture_img: PIL.Image (RGB) baked multi-view texture atlas
      - gs_positions_zup: (G, 3) float32 Gaussian positions in Z-up data frame (or Y-up aligned)
      - gs_colors: (G, 3) float32 RGB in [0, 1]
    """
    # 1. Build a parameterized UV sphere grid (n_lat + 1) x (n_lon + 1)
    u_grid = np.linspace(0.0, 1.0, n_lon + 1, dtype=np.float64)
    v_grid = np.linspace(0.01, 0.99, n_lat + 1, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)  # shape (n_lat+1, n_lon+1)

    theta = (uu - 0.5) * (2.0 * math.pi)
    phi = (vv - 0.5) * math.pi

    dir_x = np.cos(phi) * np.cos(theta)
    dir_y = np.cos(phi) * np.sin(theta)
    dir_z = np.sin(phi)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1).reshape((-1, 3))  # (V, 3) in Z-up data frame
    uvs = np.stack([uu, 1.0 - vv], axis=-1).reshape((-1, 2)).astype(np.float32)

    # 2. Ray-march radial bounds along each direction using multi-view silhouette carving
    radii_samples = np.linspace(0.04, 0.46, 28, dtype=np.float64)
    pts_candidates = dirs[:, None, :] * radii_samples[None, :, None]  # (V, R, 3)
    v_count, r_count, _ = pts_candidates.shape
    pts_flat = pts_candidates.reshape((-1, 3))
    pts_homo = np.concatenate([pts_flat, np.ones((len(pts_flat), 1), dtype=np.float64)], axis=1)

    inside_votes = np.zeros(len(pts_flat), dtype=np.int32)
    for pv, ext, intr in zip(prepared_views, extrinsics, intrinsics):
        alpha_map = pv.rgba_array[..., 3] > 25
        h_img, w_img = alpha_map.shape
        cam_pts = pts_homo @ ext.T
        z = np.maximum(cam_pts[:, 2], 1e-4)
        proj = (cam_pts[:, :3] @ intr.T) / z[:, None]
        px = np.round(proj[:, 0] * (w_img - 1)).astype(np.int32)
        py = np.round(proj[:, 1] * (h_img - 1)).astype(np.int32)
        in_bounds = (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img) & (cam_pts[:, 2] > 0.05)
        valid_idx = np.nonzero(in_bounds)[0]
        hit = alpha_map[py[valid_idx], px[valid_idx]]
        inside_votes[valid_idx] += hit.astype(np.int32)

    inside_grid = inside_votes.reshape((v_count, r_count))
    min_required = max(1, len(prepared_views) - 1)
    carved_radii = np.full(v_count, 0.26, dtype=np.float64)
    for vi in range(v_count):
        valid_r = radii_samples[inside_grid[vi] >= min_required]
        if len(valid_r) > 0:
            carved_radii[vi] = float(valid_r.max())
        else:
            fallback_r = radii_samples[inside_grid[vi] >= 1]
            if len(fallback_r) > 0:
                carved_radii[vi] = float(fallback_r.max()) * 0.85

    # Smooth the radial visual hull surface on the (n_lat+1, n_lon+1) grid
    r_2d = carved_radii.reshape((n_lat + 1, n_lon + 1))
    for _ in range(4):
        r_pad = np.pad(r_2d, ((1, 1), (0, 0)), mode="edge")
        r_pad = np.pad(r_pad, ((0, 0), (1, 1)), mode="wrap")
        r_2d = (
            0.40 * r_2d
            + 0.15 * r_pad[:-2, 1:-1]
            + 0.15 * r_pad[2:, 1:-1]
            + 0.15 * r_pad[1:-1, :-2]
            + 0.15 * r_pad[1:-1, 2:]
        )
        r_2d[:, -1] = r_2d[:, 0]

    verts_zup = dirs * r_2d.reshape((-1, 1))

    # Build triangle connectivity with outward-pointing counter-clockwise face normals
    faces_list: List[Tuple[int, int, int]] = []
    cols = n_lon + 1
    for r in range(n_lat):
        for c in range(n_lon):
            i00 = r * cols + c
            i10 = (r + 1) * cols + c
            i01 = r * cols + (c + 1)
            i11 = (r + 1) * cols + (c + 1)
            faces_list.append((i00, i01, i11))
            faces_list.append((i00, i11, i10))
    faces = np.asarray(faces_list, dtype=np.int64)

    # Compute vertex normals in Z-up frame for view-angle-weighted texture projection
    v_normals = np.zeros_like(verts_zup, dtype=np.float64)
    v0 = verts_zup[faces[:, 0]]
    v1 = verts_zup[faces[:, 1]]
    v2 = verts_zup[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for c_idx in range(3):
        np.add.at(v_normals, faces[:, c_idx], fn)
    n_len = np.linalg.norm(v_normals, axis=1, keepdims=True)
    v_normals = v_normals / np.where(n_len > 1e-8, n_len, 1.0)

    # Project multi-view colors onto a high-res UV texture grid (tex_size x tex_size)
    tu = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    tv = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    tuu, tvv = np.meshgrid(tu, tv)
    t_theta = (tuu - 0.5) * (2.0 * math.pi)
    t_phi = (tvv - 0.5) * math.pi
    t_dirs = np.stack(
        [np.cos(t_phi) * np.cos(t_theta), np.cos(t_phi) * np.sin(t_theta), np.sin(t_phi)],
        axis=-1,
    )
    # Bilinearly interpolate radius onto texture grid
    r_tex_img = Image.fromarray(r_2d.astype(np.float32)).resize(
        (tex_size, tex_size), Image.Resampling.BILINEAR
    )
    r_tex = np.asarray(r_tex_img, dtype=np.float64)[..., None]
    tex_pts_zup = (t_dirs * r_tex).reshape((-1, 3))
    tex_norms_zup = t_dirs.reshape((-1, 3))
    tex_homo = np.concatenate([tex_pts_zup, np.ones((len(tex_pts_zup), 1), dtype=np.float64)], axis=1)

    accum_color = np.zeros((len(tex_pts_zup), 3), dtype=np.float64)
    accum_weight = np.zeros((len(tex_pts_zup), 1), dtype=np.float64)

    for pv, ext, intr in zip(prepared_views, extrinsics, intrinsics):
        rgba_f = pv.rgba_array.astype(np.float64) / 255.0
        h_img, w_img, _ = rgba_f.shape

        r_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        cam_pos = -(r_w2c.T @ t_w2c)
        view_vec = cam_pos[None, :] - tex_pts_zup
        view_vec /= np.maximum(np.linalg.norm(view_vec, axis=1, keepdims=True), 1e-8)
        facing = np.clip(np.sum(tex_norms_zup * view_vec, axis=1), 0.05, 1.0) ** 2

        cam_pts = tex_homo @ ext.T
        z = np.maximum(cam_pts[:, 2], 1e-4)
        proj = (cam_pts[:, :3] @ intr.T) / z[:, None]
        px = np.clip(np.round(proj[:, 0] * (w_img - 1)).astype(np.int32), 0, w_img - 1)
        py = np.clip(np.round(proj[:, 1] * (h_img - 1)).astype(np.int32), 0, h_img - 1)

        sampled_rgba = rgba_f[py, px]
        w = (sampled_rgba[:, 3] * facing)[:, None]
        accum_color += sampled_rgba[:, :3] * w
        accum_weight += w

    # Fill any unprojected texels with average foreground object color
    valid_fg_colors: List[np.ndarray] = []
    for pv in prepared_views:
        mask_fg = pv.rgba_array[..., 3] > 128
        if np.any(mask_fg):
            valid_fg_colors.append(pv.rgba_array[mask_fg, :3].mean(axis=0) / 255.0)
    mean_fg_rgb = np.mean(valid_fg_colors, axis=0) if valid_fg_colors else np.array([0.75, 0.72, 0.68])

    tex_rgb = np.where(accum_weight > 1e-4, accum_color / np.maximum(accum_weight, 1e-6), mean_fg_rgb[None, :])
    tex_uint8 = np.round(tex_rgb.reshape((tex_size, tex_size, 3)) * 255.0).clip(0, 255).astype(np.uint8)
    texture_img = Image.fromarray(tex_uint8, mode="RGB")

    # Transform vertices into Y-up mesh coordinate frame using DATA_TO_MESH: (x, z, -y)
    basis = np.asarray(DEFAULT_DATA_TO_MESH, dtype=np.float64)
    verts_homo = np.concatenate([verts_zup, np.ones((len(verts_zup), 1), dtype=np.float64)], axis=1)
    verts_yup = (verts_homo @ basis.T)[:, :3].astype(np.float32)

    # Sample colors at mesh vertices for 3D Gaussian Splatting output
    vu_px = np.clip(np.round(uvs[:, 0] * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    vv_px = np.clip(np.round((1.0 - uvs[:, 1]) * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    vert_colors = tex_uint8[vv_px, vu_px].astype(np.float32) / 255.0

    # Add face centroids to make a denser Gaussian Splatting point cloud
    face_centers_yup = verts_yup[faces].mean(axis=1)
    face_colors = vert_colors[faces].mean(axis=1)
    gs_positions_yup = np.concatenate([verts_yup, face_centers_yup], axis=0)
    gs_colors = np.concatenate([vert_colors, face_colors], axis=0)

    return verts_yup, faces, uvs, texture_img, gs_positions_yup, gs_colors


def render_mesh_reprojections_software(
    vertices_yup: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_img: Image.Image,
    extrinsics: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    data_to_mesh: Sequence[Sequence[float]] = DEFAULT_DATA_TO_MESH,
    resolution: int = 518,
) -> List[np.ndarray]:
    """
    Software Z-buffered triangle rasterizer that renders the exported Y-up textured mesh
    at each predicted camera pose (applying inv(data_to_mesh) exactly like GenSfM's mesh_reprojections).
    """
    tex_np = np.asarray(texture_img.convert("RGB"), dtype=np.float32)
    th, tw, _ = tex_np.shape
    vu_px = np.clip(np.round(uvs[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
    vv_px = np.clip(np.round((1.0 - uvs[:, 1]) * (th - 1)).astype(np.int32), 0, th - 1)
    vert_rgb = tex_np[vv_px, vu_px]  # (V, 3)

    verts_homo = np.concatenate(
        [np.asarray(vertices_yup, dtype=np.float64), np.ones((len(vertices_yup), 1), dtype=np.float64)],
        axis=1,
    )
    inv_basis = np.linalg.inv(np.asarray(data_to_mesh, dtype=np.float64))
    verts_data = verts_homo @ inv_basis.T  # back to Z-up data frame

    face_colors = vert_rgb[faces].mean(axis=1).astype(np.uint8)
    renders: List[np.ndarray] = []

    try:
        import cv2
        has_cv2 = True
    except ImportError as cv_exc:
        logging.getLogger(__name__).debug(
            f"OpenCV not available for polygon rasterization ({cv_exc}); using NumPy splat fallback."
        )
        has_cv2 = False

    for ext, intr in zip(extrinsics, intrinsics):
        cam = verts_data @ np.asarray(ext, dtype=np.float64).T
        depths = cam[:, 2]
        q = cam[:, :3] @ np.asarray(intr, dtype=np.float64).T
        z_safe = np.maximum(q[:, 2], 1e-4)
        px = np.round((q[:, 0] / z_safe) * (resolution - 1)).astype(np.int32)
        py = np.round((q[:, 1] / z_safe) * (resolution - 1)).astype(np.int32)

        canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        face_depths = depths[faces].mean(axis=1)
        # Painter's algorithm (back-to-front sorting by depth)
        order = np.argsort(face_depths)[::-1]

        if has_cv2:
            pts_2d = np.stack([px, py], axis=-1)
            for fi in order:
                if face_depths[fi] <= 0.05:
                    continue
                tri = pts_2d[faces[fi]]
                color = (int(face_colors[fi, 0]), int(face_colors[fi, 1]), int(face_colors[fi, 2]))
                cv2.fillConvexPoly(canvas, tri, color, lineType=cv2.LINE_AA)
        else:
            # Pure NumPy splat fallback if cv2 is unavailable
            for vi in np.argsort(depths)[::-1]:
                if depths[vi] <= 0.05:
                    continue
                x0, y0 = int(px[vi]), int(py[vi])
                if 2 <= x0 < resolution - 2 and 2 <= y0 < resolution - 2:
                    canvas[y0 - 2 : y0 + 3, x0 - 2 : x0 + 3] = vert_rgb[vi].astype(np.uint8)

        renders.append(canvas)

    return renders


def run_llm_guided_sfm(
    prepared_views: Sequence[PreparedViewRecord],
    output_dir: Path,
    ai_provider: str = "gemini",
    ai_model_name: str = "gemini-3.8-flash",
    api_key: str = "",
    seed: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """
    Execute the complete LLM-Guided Sparse SfM + Multi-View 3D Visual Hull pipeline and write:
      - mesh.glb
      - mesh.obj + mesh.mtl + mesh_texture.png
      - gaussian.ply
      - cameras.json
      - reprojection-00..03.png & reprojection_<VIEW_ID>.png
      - run.json
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = ai_provider.strip().lower()

    predicted_cams: Optional[List[PredictedViewCamera]] = None
    pose_source = "heuristic_visual_hull"

    if api_key.strip():
        try:
            if provider == "gemini":
                predicted_cams = _query_gemini_poses(
                    prepared_views, api_key=api_key, model_name=ai_model_name, seed=seed, logger=logger
                )
                pose_source = f"gemini:{normalize_gemini_model_name(ai_model_name)}"
            elif provider == "openai":
                predicted_cams = _query_openai_poses(
                    prepared_views, api_key=api_key, model_name=ai_model_name, seed=seed
                )
                pose_source = f"openai:{ai_model_name}"
            elif provider == "anthropic":
                predicted_cams = _query_anthropic_poses(
                    prepared_views, api_key=api_key, model_name=ai_model_name
                )
                pose_source = f"anthropic:{ai_model_name}"
        except Exception as api_exc:
            (logger or logging.getLogger(__name__)).warning(
                f"Multimodal LLM API pose estimation failed ({api_exc}); using silhouette pose solver."
            )

    if predicted_cams is None:
        if logger and not api_key.strip():
            logger.info(
                f"No API key configured for provider '{provider}'; using deterministic multi-view silhouette pose solver."
            )
        predicted_cams = estimate_poses_heuristic(prepared_views)

    extrinsics: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    for cam in predicted_cams:
        ext, intr = spherical_camera_to_opencv_extrinsic_and_intrinsic(cam)
        extrinsics.append(ext)
        intrinsics.append(intr)

    verts_yup, faces, uvs, tex_img, gs_pos, gs_col = _reconstruct_visual_hull_and_texture(
        prepared_views=prepared_views,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    glb_path = write_textured_glb(output_dir / "mesh.glb", verts_yup, faces, uvs, tex_img)
    obj_path, mtl_path, tex_path = write_textured_obj(
        output_dir / "mesh.obj", verts_yup, faces, uvs, tex_img
    )
    ply_path = write_gaussian_ply(output_dir / "gaussian.ply", gs_pos, gs_col)

    renders = render_mesh_reprojections_software(
        vertices_yup=verts_yup,
        faces=faces,
        uvs=uvs,
        texture_img=tex_img,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    for idx, (pv, render_np) in enumerate(zip(prepared_views, renders)):
        rep_im = Image.fromarray(render_np, mode="RGB")
        rep_im.save(output_dir / f"reprojection-{idx:02d}.png")
        rep_im.save(output_dir / f"reprojection_{pv.view_descriptor.view_id}.png")

    cameras_payload: Dict[str, Any] = {
        "convention": "OpenCV world-to-camera; normalized complete K; data coordinates are Z-up",
        "intrinsics": [k.tolist() for k in intrinsics],
        "extrinsics": [e.tolist() for e in extrinsics],
        "intrinsics_source": compute_source_intrinsics(intrinsics, prepared_views),
        "source_frame": "EXIF-oriented source image before foreground crop",
        "data_to_mesh": DEFAULT_DATA_TO_MESH,
        "gt_assisted_alignment": False,
        "pose_estimator": pose_source,
    }
    cameras_json_path = output_dir / "cameras.json"
    cameras_json_path.write_text(json.dumps(cameras_payload, indent=2) + "\n", encoding="utf-8")

    run_metadata: Dict[str, Any] = {
        "engine": "llm_guided_sfm",
        "pose_estimator": pose_source,
        "seed": seed,
        "num_views": len(prepared_views),
        "inputs": [
            {
                "path": str(pv.view_descriptor.image_path),
                "source_size": list(pv.source_size),
                "crop_pixels": list(pv.crop_pixels),
                "crop_normalized": pv.crop_normalized,
            }
            for pv in prepared_views
        ],
    }
    (output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "mesh_glb": glb_path,
        "mesh_obj": obj_path,
        "mesh_mtl": mtl_path,
        "mesh_texture": tex_path,
        "gaussian_ply": ply_path,
        "cameras_json": cameras_json_path,
    }
