"""
Input View Selection & Hybrid Foreground Masking for mrGenSfM.

Supports:
- Reading AliceVision SfMData (.sfm JSON, .abc Alembic via pyalicevision/CLI) or image folders
- Selecting 2-4 views (manual_indices, uniform_spread, ai_smart_select)
- Generating RGBA foreground-segmented inputs (existing_alpha, external_masks, auto_birefnet,
  auto_rembg, auto_grabcut) with safe padding so GenSfM crop never truncates foreground pixels.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


class InsufficientViewsError(RuntimeError, ValueError):
    """Raised when fewer than 2 views are provided to GenSfM."""



@dataclass
class ViewDescriptor:
    """Metadata for a single input camera view from CameraInit or an image folder."""

    view_id: str
    pose_id: str
    intrinsic_id: str
    frame_id: int
    image_path: Path
    width: int
    height: int
    focal_px: Optional[float] = None
    sensor_width_mm: float = 36.0


@dataclass
class PreparedViewRecord:
    """Record of a prepared & masked RGBA view ready for GenSfM or LLM-SfM."""

    index: int
    view_descriptor: ViewDescriptor
    rgba_path: Path
    rgb_cond_path: Path
    gallery_mask_path: Path
    source_size: Tuple[int, int]
    crop_pixels: Tuple[int, int, int, int]
    crop_normalized: List[List[float]]
    rgba_array: np.ndarray = field(repr=False)


def _load_views_from_sfm_json(sfm_path: Path) -> List[ViewDescriptor]:
    """Parse views and intrinsics from an AliceVision .sfm / .json file."""
    data = json.loads(sfm_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "views" not in data:
        raise ValueError(f"File is not a valid AliceVision SfMData JSON: {sfm_path}")

    intrinsics_map: Dict[str, Dict[str, Any]] = {}
    for intr in data.get("intrinsics", []):
        if isinstance(intr, dict) and "intrinsicId" in intr:
            intrinsics_map[str(intr["intrinsicId"])] = intr

    views: List[ViewDescriptor] = []
    for raw_view in data.get("views", []):
        if not isinstance(raw_view, dict) or "path" not in raw_view:
            continue
        img_path = Path(str(raw_view["path"])).expanduser().resolve()
        view_id = str(raw_view.get("viewId", len(views)))
        pose_id = str(raw_view.get("poseId", view_id))
        intr_id = str(raw_view.get("intrinsicId", "0"))
        try:
            frame_id = int(raw_view.get("frameId", len(views)))
        except (ValueError, TypeError) as frame_exc:
            logging.getLogger(__name__).debug(
                f"Non-integer frameId in view {view_id} ({frame_exc}); defaulting to index."
            )
            frame_id = len(views)

        width = int(raw_view.get("width", 0) or 0)
        height = int(raw_view.get("height", 0) or 0)
        if (width <= 0 or height <= 0) and img_path.is_file():
            with Image.open(img_path) as im:
                width, height = im.size

        focal_px: Optional[float] = None
        sensor_w = 36.0
        intr_info = intrinsics_map.get(intr_id)
        if intr_info:
            try:
                sensor_w = float(intr_info.get("sensorWidth", 36.0) or 36.0)
                fl_mm = float(intr_info.get("focalLength", -1.0) or -1.0)
                if fl_mm > 0 and sensor_w > 0 and width > 0:
                    focal_px = (fl_mm / sensor_w) * float(max(width, height))
            except (ValueError, TypeError) as intr_exc:
                logging.getLogger(__name__).debug(
                    f"Could not parse intrinsic focal metadata for intrinsicId {intr_id}: {intr_exc}"
                )
                focal_px = None

        views.append(
            ViewDescriptor(
                view_id=view_id,
                pose_id=pose_id,
                intrinsic_id=intr_id,
                frame_id=frame_id,
                image_path=img_path,
                width=width,
                height=height,
                focal_px=focal_px,
                sensor_width_mm=sensor_w,
            )
        )

    views.sort(key=lambda v: (v.frame_id, v.image_path.name))
    return views


def _load_views_from_abc(abc_path: Path, logger: Optional[logging.Logger] = None) -> List[ViewDescriptor]:
    """Load views from an AliceVision .abc file using pyalicevision or aliceVision_convertSfMFormat."""
    try:
        from pyalicevision import sfmData, sfmDataIO

        data_av = sfmData.SfMData()
        if sfmDataIO.load(data_av, str(abc_path), sfmDataIO.ALL):
            views: List[ViewDescriptor] = []
            av_views = data_av.getViews()
            for vid, v in av_views.items():
                img_info = v.getImage()
                img_path = Path(img_info.getImagePath()).resolve()
                views.append(
                    ViewDescriptor(
                        view_id=str(vid),
                        pose_id=str(v.getPoseId()),
                        intrinsic_id=str(v.getIntrinsicId()),
                        frame_id=int(v.getFrameId()),
                        image_path=img_path,
                        width=int(img_info.getWidth()),
                        height=int(img_info.getHeight()),
                    )
                )
            views.sort(key=lambda item: (item.frame_id, item.image_path.name))
            return views
    except Exception as exc:
        (logger or logging.getLogger(__name__)).debug(
            f"pyalicevision direct .abc load unavailable ({exc}), trying CLI converter."
        )

    av_root = os.environ.get("ALICEVISION_ROOT", "D:/Meshroom/Meshroom-2025.1.0/aliceVision")
    converter_exe = Path(av_root) / "bin" / ("aliceVision_convertSfMFormat.exe" if os.name == "nt" else "aliceVision_convertSfMFormat")
    if converter_exe.is_file():
        with tempfile.TemporaryDirectory(prefix="gensfm_sfm_") as tmp_dir:
            tmp_sfm = Path(tmp_dir) / "converted.sfm"
            env = os.environ.copy()
            env["ALICEVISION_ROOT"] = str(av_root)
            subprocess.run(
                [str(converter_exe), "--input", str(abc_path), "--output", str(tmp_sfm)],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return _load_views_from_sfm_json(tmp_sfm)

    raise RuntimeError(f"Unable to read Alembic SfMData file: {abc_path}")


def load_input_views(input_path: str | Path, logger: Optional[logging.Logger] = None) -> List[ViewDescriptor]:
    """Load input views from an AliceVision SfMData file (.sfm/.json/.abc) or an image directory."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        img_files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        views: List[ViewDescriptor] = []
        for idx, img_file in enumerate(img_files):
            with Image.open(img_file) as im:
                w, h = im.size
            views.append(
                ViewDescriptor(
                    view_id=str(1000 + idx),
                    pose_id=str(1000 + idx),
                    intrinsic_id="100",
                    frame_id=idx,
                    image_path=img_file,
                    width=w,
                    height=h,
                )
            )
        if len(views) < 2:
            raise InsufficientViewsError(
                f"GenSfM requires at least 2 views of the object (4 random views recommended). Found {len(views)} in {path}."
            )
        return views

    suffix = path.suffix.lower()
    if suffix in (".sfm", ".json"):
        views = _load_views_from_sfm_json(path)
    elif suffix == ".abc":
        views = _load_views_from_abc(path, logger=logger)
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return load_input_views(path.parent, logger=logger)
    else:
        raise ValueError(f"Unsupported input file format: {path}")

    valid_views = [v for v in views if v.image_path.is_file()]
    if len(valid_views) < 2:
        raise InsufficientViewsError(
            f"GenSfM requires at least 2 views of the object (4 random views recommended). Found {len(valid_views)} in {path}."
        )
    return valid_views


def _score_view_quality(image_path: Path) -> Tuple[float, np.ndarray]:
    """Compute a sharpness + foreground centrality score and color histogram for smart view selection."""
    with Image.open(image_path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((256, 256)), dtype=np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Discrete Laplacian for sharpness
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(np.var(lap))
    hist, _ = np.histogram(rgb, bins=16, range=(0, 256), density=True)
    return sharpness, hist.astype(np.float32)


def select_views(
    views: Sequence[ViewDescriptor],
    selection_mode: str = "all_up_to_4",
    indices_str: str = "0,1,2,3",
    max_views: int = 4,
    logger: Optional[logging.Logger] = None,
) -> List[ViewDescriptor]:
    """Select 2 to 4 random/sparse views from the input sequence."""
    total = len(views)
    if total < 2:
        raise InsufficientViewsError(
            f"GenSfM requires at least 2 views of the object (4 random views recommended). Got {total}."
        )

    if total > max_views and logger:
        logger.warning(
            f"CameraInit contains {total} views (> {max_views}). Automatically selecting {max_views} views "
            f"using viewSelectionMode='{selection_mode}'."
        )

    if total <= max_views and selection_mode != "manual_indices":
        return list(views)

    target_count = min(max_views, total)

    if selection_mode == "all_up_to_4":
        selected = list(views[:target_count])
        if logger:
            logger.info(f"Sequential up-to-4 view selection chose: {[v.image_path.name for v in selected]}")
        return selected

    if selection_mode == "uniform_spread":
        chosen_indices = [int(round(i * (total - 1) / (target_count - 1))) for i in range(target_count)]
        # Deduplicate while preserving order
        seen = set()
        unique_indices: List[int] = []
        for idx in chosen_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        return [views[i] for i in unique_indices]

    if selection_mode in ("auto_diverse", "ai_smart_select"):
        scored: List[Tuple[int, float, np.ndarray]] = []
        for idx, v in enumerate(views):
            sharpness, hist = _score_view_quality(v.image_path)
            scored.append((idx, sharpness, hist))
        # Pick sharpest first, then greedily maximize viewpoint appearance diversity + sharpness
        scored.sort(key=lambda item: item[1], reverse=True)
        selected_idx = [scored[0][0]]
        selected_hists = [scored[0][2]]
        while len(selected_idx) < target_count:
            best_candidate = -1
            best_metric = -1.0
            for idx, sharp, hist in scored:
                if idx in selected_idx:
                    continue
                min_dist = min(float(np.linalg.norm(hist - sh)) for sh in selected_hists)
                metric = min_dist * math.log1p(max(sharp, 1.0))
                if metric > best_metric:
                    best_metric = metric
                    best_candidate = idx
            if best_candidate < 0:
                break
            selected_idx.append(best_candidate)
            for idx, _, hist in scored:
                if idx == best_candidate:
                    selected_hists.append(hist)
                    break
        selected_idx.sort()
        if logger:
            logger.info(f"Diverse view selection ({selection_mode}) chose view indices: {selected_idx}")
        return [views[i] for i in selected_idx]

    # Default: manual_indices
    parsed: List[int] = []
    for token in indices_str.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
            if 0 <= idx < total and idx not in parsed:
                parsed.append(idx)
        except ValueError as idx_exc:
            (logger or logging.getLogger(__name__)).debug(
                f"Ignoring invalid view index token '{token}': {idx_exc}"
            )
            continue

    if len(parsed) < 2:
        parsed = list(range(min(target_count, total)))
    else:
        parsed = parsed[:max_views]

    return [views[i] for i in parsed]


def _segment_with_grabcut_or_border(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Segment the foreground object from an RGB uint8 image.
    Uses OpenCV GrabCut when cv2 is available, with an adaptive border-color distance fallback.
    Returns an alpha mask in uint8 [0..255].
    """
    h, w, _ = rgb_uint8.shape
    try:
        import cv2

        mask = np.zeros((h, w), dtype=np.uint8)
        pad_x = max(4, int(w * 0.06))
        pad_y = max(4, int(h * 0.06))
        rect = (pad_x, pad_y, max(1, w - 2 * pad_x), max(1, h - 2 * pad_y))
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(rgb_uint8, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if np.count_nonzero(fg_mask) > int(0.02 * h * w):
            kernel = np.ones((3, 3), dtype=np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
            return fg_mask
    except Exception as grabcut_exc:
        logging.getLogger(__name__).debug(
            f"OpenCV GrabCut segmentation unavailable or failed ({grabcut_exc}); using border-color fallback."
        )

    # Pure NumPy border color distance fallback
    border_pixels = np.concatenate(
        [
            rgb_uint8[0, :, :],
            rgb_uint8[-1, :, :],
            rgb_uint8[:, 0, :],
            rgb_uint8[:, -1, :],
        ],
        axis=0,
    ).astype(np.float32)
    bg_median = np.median(border_pixels, axis=0)
    diff = np.linalg.norm(rgb_uint8.astype(np.float32) - bg_median[None, None, :], axis=-1)
    threshold = max(18.0, float(np.percentile(diff, 35)))
    alpha = np.where(diff > threshold, 255, 0).astype(np.uint8)
    if np.count_nonzero(alpha) < int(0.02 * h * w):
        # Central ellipse fallback if background is indistinguishable
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.38, w * 0.38
        ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        alpha = np.where(ellipse, 255, 0).astype(np.uint8)
    return alpha


def _apply_masking_to_image(
    image_rgba: Image.Image,
    view: ViewDescriptor,
    masking_mode: str,
    input_masks_folder: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> Image.Image:
    """Apply the selected foreground masking mode to produce a clean RGBA PIL Image."""
    rgba_np = np.asarray(image_rgba, dtype=np.uint8).copy()
    rgb_np = rgba_np[..., :3]
    alpha_np = rgba_np[..., 3]
    has_existing_transparency = bool(np.any(alpha_np < 250) and np.any(alpha_np > 5))

    if masking_mode == "existing_alpha" and has_existing_transparency:
        return Image.fromarray(rgba_np, mode="RGBA")

    if masking_mode == "external_masks" and input_masks_folder and input_masks_folder.is_dir():
        candidates = [
            input_masks_folder / f"{view.image_path.stem}.png",
            input_masks_folder / f"{view.image_path.stem}.exr",
            input_masks_folder / f"{view.view_id}.png",
            input_masks_folder / f"{view.view_id}.exr",
            input_masks_folder / f"mask_{view.view_id}.png",
        ]
        for cand in candidates:
            if cand.is_file() and cand.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                with Image.open(cand) as mask_im:
                    mask_gray = mask_im.convert("L").resize(image_rgba.size, Image.Resampling.BILINEAR)
                    rgba_np[..., 3] = np.asarray(mask_gray, dtype=np.uint8)
                    return Image.fromarray(rgba_np, mode="RGBA")
        if logger:
            logger.warning(f"No matching mask found in {input_masks_folder} for {view.image_path.name}, falling back.")

    if has_existing_transparency and masking_mode in ("auto_rembg", "auto_birefnet", "existing_alpha"):
        return Image.fromarray(rgba_np, mode="RGBA")

    if masking_mode == "auto_birefnet":
        try:
            from segmentationRDS import segmentation

            processor = segmentation.BiRefNetSeg(modelType="BiRefNet LR", useGPU=True)
            bboxes = np.asarray([[0, 0, rgb_np.shape[1] - 1, rgb_np.shape[0] - 1]])
            mask_float = processor.process(
                image=rgb_np.astype(np.float32) / 255.0,
                bboxes=bboxes,
                invert=False,
                verbose=False,
            )
            alpha_pred = np.clip(np.squeeze(mask_float) * 255.0, 0, 255).astype(np.uint8)
            rgba_np[..., 3] = alpha_pred
            return Image.fromarray(rgba_np, mode="RGBA")
        except Exception as exc:
            (logger or logging.getLogger(__name__)).info(
                f"BiRefNet local matting unavailable ({exc}), falling back to GrabCut/rembg."
            )

    if masking_mode == "auto_rembg":
        try:
            import rembg

            out_im = rembg.remove(image_rgba).convert("RGBA")
            out_alpha = np.asarray(out_im.getchannel("A"))
            if np.any(out_alpha > 10):
                return out_im
        except Exception as exc:
            (logger or logging.getLogger(__name__)).debug(
                f"rembg package not installed locally ({exc}), using GrabCut segmentation."
            )

    rgba_np[..., 3] = _segment_with_grabcut_or_border(rgb_np)
    return Image.fromarray(rgba_np, mode="RGBA")


def _ensure_safe_foreground_padding(
    image_rgba: Image.Image, crop_ratio: float = 1.2
) -> Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Ensure the RGBA image has sufficient transparent border padding so that GenSfM's
    square crop (crop_size_ratio=1.2) never truncates any foreground pixel.
    Returns (padded_or_original_image, crop_bbox_pixels, (pad_left, pad_top)).
    """
    rgba = np.asarray(image_rgba, dtype=np.uint8)
    alpha = rgba[..., 3]
    nonzero_y, nonzero_x = np.nonzero(alpha > 0)
    if len(nonzero_x) == 0 or len(nonzero_y) == 0:
        # Guard against completely empty mask
        h, w = alpha.shape
        rgba[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 3] = 255
        image_rgba = Image.fromarray(rgba, mode="RGBA")
        nonzero_y, nonzero_x = np.nonzero(rgba[..., 3] > 0)

    x_min, x_max = int(nonzero_x.min()), int(nonzero_x.max())
    y_min, y_max = int(nonzero_y.min()), int(nonzero_y.max())
    fg_w = x_max - x_min + 1
    fg_h = y_max - y_min + 1
    box_size = int(round(max(fg_w, fg_h) * crop_ratio))
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5

    left = int(round(cx - box_size * 0.5))
    top = int(round(cy - box_size * 0.5))
    right = left + box_size
    bottom = top + box_size

    w, h = image_rgba.size
    pad_left = 0
    pad_top = 0
    if left < 0 or top < 0 or right > w or bottom > h:
        pad_left = max(0, -left + 8)
        pad_top = max(0, -top + 8)
        pad_right = max(0, right - w + 8)
        pad_bottom = max(0, bottom - h + 8)
        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom
        padded = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        padded.paste(image_rgba, (pad_left, pad_top))
        image_rgba = padded
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
        w, h = image_rgba.size

    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    return image_rgba, (left, top, right, bottom), (pad_left, pad_top)


def prepare_masked_views(
    views: Sequence[ViewDescriptor],
    output_dir: Path,
    masking_mode: str = "auto_rembg",
    input_masks_folder: str = "",
    crop_ratio: float = 1.2,
    target_resolution: int = 518,
    logger: Optional[logging.Logger] = None,
) -> List[PreparedViewRecord]:
    """
    Prepare, segment, crop, and save RGBA + RGB condition images for 2-4 selected views.
    Generates:
      - {output_dir}/input-rgba-{i:02d}.png
      - {output_dir}/input-{i:02d}.png
      - {output_dir}/mask_{view_id}.png (for Meshroom 2D Viewer active viewpoint sync)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(input_masks_folder).resolve() if input_masks_folder.strip() else None

    prepared: List[PreparedViewRecord] = []
    for idx, view in enumerate(views):
        with Image.open(view.image_path) as raw_im:
            oriented = ImageOps.exif_transpose(raw_im).convert("RGBA")

        w_orig, h_orig = oriented.size
        view.width = w_orig
        view.height = h_orig
        masked_rgba = _apply_masking_to_image(
            image_rgba=oriented,
            view=view,
            masking_mode=masking_mode,
            input_masks_folder=masks_dir,
            logger=logger,
        )
        safe_rgba, bbox, pad_offsets = _ensure_safe_foreground_padding(masked_rgba, crop_ratio=crop_ratio)
        cropped = safe_rgba.crop(bbox).resize(
            (target_resolution, target_resolution), Image.Resampling.LANCZOS
        )

        # Ensure cropped RGBA still has a small transparent border (2px) so native InputViews
        # (crop_allow_padding=False, crop_size_ratio=1.2) never raises a crop truncation error
        cropped_np = np.asarray(cropped, dtype=np.uint8).copy()
        if cropped_np.shape[0] > 8 and cropped_np.shape[1] > 8:
            # Check if foreground touches outer 12% margin; if so, inset the object slightly
            alpha_c = cropped_np[..., 3]
            ys_c, xs_c = np.nonzero(alpha_c > 0)
            if len(xs_c) > 0 and len(ys_c) > 0:
                fg_span = max(int(xs_c.max() - xs_c.min() + 1), int(ys_c.max() - ys_c.min() + 1))
                max_safe_span = int(target_resolution / max(crop_ratio, 1.05)) - 4
                if fg_span > max_safe_span > 16:
                    scale_factor = float(max_safe_span) / float(fg_span)
                    inset_size = max(16, int(round(target_resolution * scale_factor)))
                    shrunk = cropped.resize((inset_size, inset_size), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGBA", (target_resolution, target_resolution), (0, 0, 0, 0))
                    offset = (target_resolution - inset_size) // 2
                    canvas.paste(shrunk, (offset, offset))
                    cropped = canvas
                    cropped_np = np.asarray(cropped, dtype=np.uint8).copy()

        alpha_f = cropped_np[..., 3:4].astype(np.float32) / 255.0
        rgb_premult = np.round(cropped_np[..., :3].astype(np.float32) * alpha_f).clip(0, 255).astype(np.uint8)

        rgba_path = output_dir / f"input-rgba-{idx:02d}.png"
        rgb_cond_path = output_dir / f"input-{idx:02d}.png"
        gallery_mask_path = output_dir / f"mask_{view.view_id}.png"
        prepared_inputs_dir = output_dir / "prepared_inputs"
        prepared_inputs_dir.mkdir(parents=True, exist_ok=True)
        prepared_rgba_path = prepared_inputs_dir / f"{idx:02d}.png"

        cropped.save(rgba_path)
        cropped.save(prepared_rgba_path)
        Image.fromarray(rgb_premult, mode="RGB").save(rgb_cond_path)
        masked_rgba.save(gallery_mask_path)

        pad_left, pad_top = pad_offsets
        orig_left = bbox[0] - pad_left
        orig_top = bbox[1] - pad_top
        orig_right = bbox[2] - pad_left
        orig_bottom = bbox[3] - pad_top
        crop_norm = [
            [float(orig_left) / float(max(1, w_orig)), float(orig_top) / float(max(1, h_orig))],
            [float(orig_right) / float(max(1, w_orig)), float(orig_bottom) / float(max(1, h_orig))],
        ]

        prepared.append(
            PreparedViewRecord(
                index=idx,
                view_descriptor=view,
                rgba_path=rgba_path,
                rgb_cond_path=rgb_cond_path,
                gallery_mask_path=gallery_mask_path,
                source_size=(w_orig, h_orig),
                crop_pixels=(orig_left, orig_top, orig_right, orig_bottom),
                crop_normalized=crop_norm,
                rgba_array=cropped_np,
            )
        )

    return prepared
