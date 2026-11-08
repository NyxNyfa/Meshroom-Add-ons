"""
3D Mesh (.glb <-> .obj + .mtl + texture.png) and 3D Gaussian Splatting (.ply) Converter for mrGenSfM.

Provides:
- Self-contained binary glTF 2.0 (.glb) reader & writer (zero external C++ dependencies required)
- Automatic conversion of mesh.glb into mesh.obj + mesh.mtl + mesh_texture.png so AliceVision
  Meshroom's built-in Qt3D SceneLoader always displays the textured mesh in the 3D Viewer
- Standard 3D Gaussian Splatting binary little-endian .ply writer and validator.
"""
from __future__ import annotations

import io
import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def write_textured_glb(
    glb_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_image: Image.Image,
) -> Path:
    """
    Export a textured triangle mesh to a standard binary glTF 2.0 (.glb) file.
    Coordinates are stored directly as provided (Y-up convention for glTF/GenSfM).
    """
    glb_path = Path(glb_path).resolve()
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    verts_f32 = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs_f32 = np.ascontiguousarray(uvs, dtype=np.float32)
    # glTF UV origin is top-left (v = 1 - v_obj)
    uvs_gltf = uvs_f32.copy()
    uvs_gltf[:, 1] = 1.0 - uvs_gltf[:, 1]
    faces_u32 = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)

    png_buf = io.BytesIO()
    texture_image.convert("RGBA").save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
        rem = len(data) % 4
        return data if rem == 0 else data + pad_byte * (4 - rem)

    v_bytes = _pad4(verts_f32.tobytes())
    uv_bytes = _pad4(uvs_gltf.tobytes())
    idx_bytes = _pad4(faces_u32.tobytes())
    img_bytes = _pad4(png_bytes)

    offset_v = 0
    offset_uv = offset_v + len(v_bytes)
    offset_idx = offset_uv + len(uv_bytes)
    offset_img = offset_idx + len(idx_bytes)
    bin_chunk = v_bytes + uv_bytes + idx_bytes + img_bytes

    v_min = verts_f32.min(axis=0).tolist() if len(verts_f32) > 0 else [-0.5, -0.5, -0.5]
    v_max = verts_f32.max(axis=0).tolist() if len(verts_f32) > 0 else [0.5, 0.5, 0.5]

    gltf_dict: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "mrGenSfM Meshroom Plugin"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "GenSfMObject"}],
        "meshes": [
            {
                "name": "GenSfMMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "GenSfMMaterial",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "doubleSided": True,
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [
            {
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }
        ],
        "images": [{"bufferView": 3, "mimeType": "image/png", "name": "mesh_texture"}],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(verts_f32)),
                "type": "VEC3",
                "min": [float(x) for x in v_min],
                "max": [float(x) for x in v_max],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(uvs_gltf)),
                "type": "VEC2",
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(len(faces_u32)),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_v, "byteLength": verts_f32.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": offset_uv, "byteLength": uvs_gltf.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": faces_u32.nbytes, "target": 34963},
            {"buffer": 0, "byteOffset": offset_img, "byteLength": len(png_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_chunk)}],
    }

    json_bytes = _pad4(json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8"), pad_byte=b" ")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)

    with open(glb_path, "wb") as f:
        # GLB Header: magic 'glTF' (0x46546C67), version 2, total_length
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk (type 0x4E4F534A)
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN chunk (type 0x004E4942)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)

    return glb_path


def write_textured_obj(
    obj_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    texture_image: Optional[Image.Image] = None,
    vertex_Normals: Optional[np.ndarray] = None,
) -> Tuple[Path, Path, Path]:
    """
    Write a textured Wavefront .obj + .mtl + mesh_texture.png bundle compatible with
    AliceVision Meshroom's 3D Viewer.
    """
    obj_path = Path(obj_path).resolve()
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")
    tex_path = obj_path.parent / "mesh_texture.png"

    if texture_image is None:
        texture_image = Image.new("RGB", (256, 256), (200, 200, 205))
    texture_image.convert("RGB").save(tex_path, format="PNG")

    mtl_content = (
        "# Generated by mrGenSfM for AliceVision Meshroom\n"
        "newmtl GenSfMMaterial\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.050000 0.050000 0.050000\n"
        "Ns 10.000000\n"
        "illum 2\n"
        f"map_Kd {tex_path.name}\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    has_uvs = uvs is not None and len(uvs) == len(verts)

    # Compute vertex normals if not supplied
    if vertex_Normals is None and len(verts) > 0 and len(tris) > 0:
        normals = np.zeros_like(verts, dtype=np.float64)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for c in range(3):
            np.add.at(normals, tris[:, c], fn)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        vertex_Normals = normals / norms

    lines: List[str] = [
        "# Generated by mrGenSfM for AliceVision Meshroom",
        f"mtllib {mtl_path.name}",
        "o GenSfMObject",
        "usemtl GenSfMMaterial",
    ]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if has_uvs and uvs is not None:
        for uv in uvs:
            lines.append(f"vt {float(uv[0]):.6f} {float(uv[1]):.6f}")

    if vertex_Normals is not None and len(vertex_Normals) == len(verts):
        for vn in vertex_Normals:
            lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")

    has_vn = vertex_Normals is not None and len(vertex_Normals) == len(verts)
    for f_idx in tris:
        i0, i1, i2 = int(f_idx[0]) + 1, int(f_idx[1]) + 1, int(f_idx[2]) + 1
        if has_uvs and has_vn:
            lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
        elif has_uvs:
            lines.append(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}")
        else:
            lines.append(f"f {i0} {i1} {i2}")

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return obj_path, mtl_path, tex_path


def _parse_glb_native(glb_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Image.Image]]:
    """Parse vertices, faces, UVs, and embedded texture from a binary glTF 2.0 (.glb) file."""
    raw = glb_path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"GLB file is truncated: {glb_path}")
    magic, version, _ = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"Invalid GLB header in {glb_path}")

    offset = 12
    json_data: Optional[Dict[str, Any]] = None
    bin_chunk = b""
    while offset + 8 <= len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_bytes = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_data = json.loads(chunk_bytes.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk_bytes

    if not json_data or not bin_chunk:
        raise ValueError(f"Missing JSON or BIN chunk in {glb_path}")

    accessors = json_data.get("accessors", [])
    buffer_views = json_data.get("bufferViews", [])
    primitive = json_data["meshes"][0]["primitives"][0]
    attrs = primitive["attributes"]

    def _read_accessor(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = buffer_views[acc["bufferView"]]
        byte_offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        count = int(acc["count"])
        comp_type = int(acc["componentType"])
        acc_type = str(acc["type"])
        num_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc_type]
        dtype_map = {
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dt = dtype_map[comp_type]
        byte_len = count * num_comp * np.dtype(dt).itemsize
        arr = np.frombuffer(bin_chunk, dtype=dt, count=count * num_comp, offset=byte_offset)
        return arr.reshape((count, num_comp)) if num_comp > 1 else arr

    vertices = _read_accessor(int(attrs["POSITION"])).astype(np.float32)
    uvs: Optional[np.ndarray] = None
    if "TEXCOORD_0" in attrs:
        uvs = _read_accessor(int(attrs["TEXCOORD_0"])).astype(np.float32).copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]

    indices = _read_accessor(int(primitive["indices"])).astype(np.int64)
    faces = indices.reshape((-1, 3))

    tex_image: Optional[Image.Image] = None
    images_list = json_data.get("images", [])
    if images_list and "bufferView" in images_list[0]:
        img_bv = buffer_views[int(images_list[0]["bufferView"])]
        img_off = int(img_bv.get("byteOffset", 0))
        img_len = int(img_bv["byteLength"])
        tex_image = Image.open(io.BytesIO(bin_chunk[img_off : img_off + img_len])).convert("RGB")

    return vertices, faces, uvs, tex_image


def convert_glb_to_obj(
    glb_path: Path,
    obj_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Path, Path]:
    """
    Convert a GenSfM mesh.glb file into mesh.obj + mesh.mtl + mesh_texture.png.
    Uses native binary GLB parsing with optional trimesh fallback for complex multi-node GLBs.
    """
    glb_path = Path(glb_path).resolve()
    obj_path = Path(obj_path).resolve()
    if not glb_path.is_file():
        raise FileNotFoundError(f"GLB file does not exist: {glb_path}")

    try:
        vertices, faces, uvs, tex_image = _parse_glb_native(glb_path)
        return write_textured_obj(obj_path, vertices, faces, uvs, tex_image)
    except Exception as native_exc:
        (logger or logging.getLogger(__name__)).debug(
            f"Native GLB parser encountered non-standard layout ({native_exc}), trying trimesh."
        )

    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), force="mesh")
    vertices = np.asarray(scene_or_mesh.vertices, dtype=np.float32)
    faces = np.asarray(scene_or_mesh.faces, dtype=np.int64)
    uvs = None
    tex_image = None
    if hasattr(scene_or_mesh, "visual"):
        if hasattr(scene_or_mesh.visual, "uv") and scene_or_mesh.visual.uv is not None:
            uvs = np.asarray(scene_or_mesh.visual.uv, dtype=np.float32)
        mat = getattr(scene_or_mesh.visual, "material", None)
        if mat is not None and getattr(mat, "baseColorTexture", None) is not None:
            tex_image = mat.baseColorTexture.convert("RGB")

    return write_textured_obj(obj_path, vertices, faces, uvs, tex_image)


def write_gaussian_ply(
    ply_path: Path,
    positions: np.ndarray,
    colors_rgb_01: np.ndarray,
    scales: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
) -> Path:
    """
    Write a 3D Gaussian Splatting .ply file in binary_little_endian format.
    Includes both standard point cloud RGB (red, green, blue) for Qt3D SceneLoader compatibility
    and standard 3DGS spherical harmonics DC (f_dc_0..2), opacity, scale_0..2, and rot_0..3
    for mrGSplat compatibility.
    """
    ply_path = Path(ply_path).resolve()
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.ascontiguousarray(positions, dtype=np.float32)
    n_pts = len(pts)
    rgb = np.clip(np.ascontiguousarray(colors_rgb_01, dtype=np.float32), 0.0, 1.0)
    rgb_u8 = np.round(rgb * 255.0).clip(0, 255).astype(np.uint8)

    # Convert RGB [0, 1] to 0th-order Spherical Harmonics (C0 = 0.28209479177387814)
    c0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / c0

    if opacities is None:
        # Inverse sigmoid of ~0.92 opacity -> 2.44
        opac = np.full((n_pts, 1), 2.44, dtype=np.float32)
    else:
        opac = np.ascontiguousarray(opacities, dtype=np.float32).reshape((n_pts, 1))

    if scales is None:
        # Log-scale default (~0.015 world units -> log(0.015) = -4.2)
        sc = np.full((n_pts, 3), -4.2, dtype=np.float32)
    else:
        sc = np.ascontiguousarray(scales, dtype=np.float32).reshape((n_pts, 3))

    rot = np.zeros((n_pts, 4), dtype=np.float32)
    rot[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )

    structured = np.zeros(n_pts, dtype=vertex_dtype)
    structured["x"] = pts[:, 0]
    structured["y"] = pts[:, 1]
    structured["z"] = pts[:, 2]
    structured["red"] = rgb_u8[:, 0]
    structured["green"] = rgb_u8[:, 1]
    structured["blue"] = rgb_u8[:, 2]
    structured["f_dc_0"] = f_dc[:, 0]
    structured["f_dc_1"] = f_dc[:, 1]
    structured["f_dc_2"] = f_dc[:, 2]
    structured["opacity"] = opac[:, 0]
    structured["scale_0"] = sc[:, 0]
    structured["scale_1"] = sc[:, 1]
    structured["scale_2"] = sc[:, 2]
    structured["rot_0"] = rot[:, 0]
    structured["rot_1"] = rot[:, 1]
    structured["rot_2"] = rot[:, 2]
    structured["rot_3"] = rot[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated by mrGenSfM for AliceVision Meshroom & mrGSplat\n"
        f"element vertex {n_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(structured.tobytes())

    return ply_path
