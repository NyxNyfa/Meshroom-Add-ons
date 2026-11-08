# mrGenSfM — Sparse Random-View AI Initialization & 3D Reconstruction Plugin for AliceVision Meshroom

`mrGenSfM` transforms the input requirement of **[AliceVision Meshroom 2025.1.0+](https://github.com/alicevision/Meshroom)** from **30–150+ sequential high-overlap photos** down to just **2 to 4 uncalibrated photos taken from random angles** using **[GenSfM: Generative Structure from Motion with Native 3D Diffusion](https://github.com/dotb1ue/GenSfm)** ([arXiv:2610.35522](https://arxiv.org/abs/2610.35522)) and a **Multimodal LLM-Guided Sparse SfM + Visual Hull Engine** inside a single modular node (`GenSfM`).

---

## Standard Meshroom vs. Meshroom + `GenSfM` Node

| Input & Processing Parameter | Standard Meshroom Pipeline (`photogrammetry.mg`) | Meshroom + `GenSfM` Pipeline (`genSfm.mg`) |
| :--- | :--- | :--- |
| **Number of Photos in `CameraInit`** | 30 – 150+ sequential photos around the object | **Only 2 – 4 photos** from random viewpoints |
| **Overlap Requirement** | High (>60% shared area between adjacent photos) | **Free / Random** (wide baseline 90°–180° supported) |
| **Camera Calibration (EXIF / SensorDB)** | Heavily relies on camera sensor database | **Uncalibrated views supported** (intrinsics & poses jointly estimated) |
| **Occluded / Unseen Object Regions** | Leaves holes or fails meshing | Completed using 3D object-level priors or multi-view visual hull + hole-filling texturing |
| **Local Laptop Compute Load** | Heavy local CPU & GPU for tens of minutes | **Lightweight on Windows/Laptop** via `cloud_api` offload (or local WSL2/Docker/Venv) |

---

## Complete Downstream Pipeline Architecture After `GenSfM`

To achieve the same high-quality final output as standard multi-photo Meshroom photogrammetry (`photogrammetry.mg`) while letting you compare direct AI reconstruction against classic Multi-View Stereo (MVS), `mrGenSfM` provides a **Unified Dual-Branch Hybrid Pipeline Template (`genSfmHybrid.mg`)** as well as standalone templates (`genSfm.mg` and `genSfmDensePhotogrammetry.mg`) in `D:\Meshroom\mrGenSfM\meshroom\`:

### Unified Hybrid Pipeline: Dual-Branch Fast AI Polish + Dense 360-deg MVS (`genSfmHybrid.mg`) — *Recommended*

```text
CameraInit_1 (2–4 random photos)
     │
     ▼
  GenSfM_1 (generateSyntheticViews = 24 by default)
     │
     ├──► [FAST BRANCH: Direct AI Polish]
     │       ├──(outputSfMData: sfmData.abc)──► PrepareDenseScene_Fast ──┐
     │       └──(outputMeshObj: mesh.obj)─────► MeshFiltering_Fast ──────┴──► Texturing_Fast ──┐
     │                                                                                         ▼
     └──► [DENSE BRANCH: Sparse-to-Dense 28-View MVS]                                 Publish_Comparison
             └──(outputSfMDataDense: sfmDataDense.abc + 24 synthetic views)                    ▲
                     └──► PrepareDenseScene_Dense ──► DepthMap_Dense ──► DepthMapFilter_Dense  │
                                                             │                                 │
                                                             ▼                                 │
                                                       Meshing_Dense ──► MeshFiltering_Dense ──► Texturing_Dense
```

| Branch | Node Name | Input Connection(s) | Function & Role in Quality Output |
| :--- | :--- | :--- | :--- |
| **Input** | `CameraInit_1` | 2 to 4 random object photos | Initializes image viewpoints (`cameraInit.sfm`). |
| **AI Core** | `GenSfM_1` | `input`: `{CameraInit_1.output}`<br>`generateSyntheticViews`: `24` (default) | Simultaneously outputs:<br>• **Fast Branch inputs**: calibrated sparse poses (`outputSfMData` = `sfmData.abc`), raw 3D geometry (`outputMeshObj` = `mesh.obj`, `outputMeshGlb` = `mesh.glb`), and 3D Gaussian Splats (`outputGaussianPly` = `gaussian.ply`).<br>• **Dense Branch inputs**: 24 orbital 360-deg multi-elevation synthetic views (`15°`, `32°`, `-8°`) combined with the 4 real views in `outputSfMDataDense` (`sfmDataDense.abc`). |
| **Fast Branch** | `PrepareDenseScene_Fast` | `input`: `{GenSfM_1.outputSfMData}` | Undistorts the 2–4 original high-resolution photos into `.exr` images (`<VIEW_ID>.exr`). |
| **Fast Branch** | `MeshFiltering_Fast` | `inputMesh`: `{GenSfM_1.outputMeshObj}` | Applies Laplacian surface smoothing (`smoothingIterations=5`) and removes stray triangles (`keepLargestMeshOnly=True`). |
| **Fast Branch** | `Texturing_Fast` | `input`: `{GenSfM_1.outputSfMData}`<br>`imagesFolder`: `{PrepareDenseScene_Fast.output}`<br>`inputMesh`: `{MeshFiltering_Fast.outputMesh}` | Unwraps UV charts and performs AliceVision **multi-band frequency blending** (`4096px`) with automatic exposure compensation (`correctEV=True`) and hole filling (`fillHoles=True`). |
| **Dense Branch** | `PrepareDenseScene_Dense` | `input`: `{GenSfM_1.outputSfMDataDense}` | Undistorts all 28 views (4 real + 24 synthetic orbital views). |
| **Dense Branch** | `DepthMap_Dense` $\rightarrow$ `DepthMapFilter_Dense` $\rightarrow$ `Meshing_Dense` $\rightarrow$ `MeshFiltering_Dense` $\rightarrow$ `Texturing_Dense` | Connected sequentially from `PrepareDenseScene_Dense` | Runs classic AliceVision PatchMatch SGM depth estimation, depth consistency filtering, 3D Delaunay tetrahedral meshing, surface smoothing, and 28-view 4K UV texturing. |
| **Comparison** | `Publish_Comparison` | `Texturing_Fast.*`, `Texturing_Dense.*`, `GenSfM_1.outputMeshGlb`, `outputGaussianPly`, `outputSfMDataDense` | Collects both branches' final meshes, `.mtl` materials, and 4K textures alongside native `.glb`, `.ply`, and `sfmDataDense.abc` for side-by-side inspection. |

---

### Standalone Pipeline 1: Direct AI Geometry + AliceVision Post-Processing (`genSfm.mg`)

```text
CameraInit_1 (2–4 random photos)
     │
     ▼
  GenSfM_1  ──(outputSfMData: sfmData.abc)──► PrepareDenseScene_1 (undistorted EXR images) ──┐
     │                                                                                       ▼
     └────────(outputMeshObj: mesh.obj)─────► MeshFiltering_1 (Laplacian smoothing) ──► Texturing_1 ──► Publish_1
```

---

### Standalone Pipeline 2: Sparse-to-Dense View Synthesis + Full Classic MVS (`genSfmDensePhotogrammetry.mg`)

```text
CameraInit_1 ──► GenSfM_1 (generateSyntheticViews=24) ──► PrepareDenseScene_1 ──► DepthMap_1 ──► DepthMapFilter_1 ──► Meshing_1 ──► MeshFiltering_1 ──► Texturing_1 ──► Publish_1
```

---

## Installation (Windows & Linux)

### 1. Register Plugin in Meshroom 2025.1.0 (Windows)
Run the included PowerShell registration script to link `D:\Meshroom\mrGenSfM` into `D:\Meshroom\Meshroom-2025.1.0\plugins\mrGenSfM`:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\Meshroom\mrGenSfM\scripts\register_plugin_windows.ps1"
```

Alternatively, add `D:\Meshroom\mrGenSfM` to `MESHROOM_PLUGINS_PATH`:
```powershell
$env:MESHROOM_PLUGINS_PATH = "D:\Meshroom\mrGenSfM"
```

### 2. Configure API Keys (`.env`)
Copy `.env.example` to `.env` and fill in any API keys you wish to use (`GENSFM_API_KEY`, `GENSFM_CLOUD_URL`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`):

```powershell
Copy-Item "D:\Meshroom\mrGenSfM\.env.example" "D:\Meshroom\mrGenSfM\.env"
```

### 3. (Optional) Setup Native GenSfM in WSL2 Ubuntu for Local GPU
```bash
bash /mnt/d/Meshroom/mrGenSfM/docker/wsl_setup_gensfm.sh /mnt/d/Meshroom/GenSfm
```

---

## Usage & 3D Viewport Navigation Guide in Meshroom 2025.1.0

1. Launch `D:\Meshroom\Meshroom-2025.1.0\Meshroom.exe`.
2. Open the pre-computed Dual-Branch Hybrid demo project `D:\Meshroom\mrGenSfM_Hybrid_Demo.mg` (or `D:\Meshroom\mrGenSfM_Wagon_Demo.mg`), **or** start a new project from `File -> New Pipeline -> genSfmHybrid` and drag 2 to 4 random photos into the **Image Gallery**.
3. Click **Start** (or run headless via `D:\Meshroom\Meshroom-2025.1.0\meshroom_compute.exe`).
4. **How to View & Compare 3D Models in the 3D Viewport (Why Viewport Starts Empty)**:
   - By design, Meshroom does **not** automatically load heavy 3D meshes into GPU memory when you first open a `.mg` project file.
   - **Double-click `Texturing_Fast`** in the Graph Editor to load the **Fast Branch** multi-band textured 3D mesh (`texturedMesh.obj` + `texture_1001.png`) into the **3D Viewer**.
   - **Double-click `GenSfM_1`** to inspect the raw AI 3D mesh (`mesh.obj`), sparse camera frustums (`sfmData.abc`), or the **24 orbital synthetic camera frustums** (`sfmDataDense.abc`).
   - **Double-click `Texturing_Dense`** (after computing the Dense MVS branch) to load the **Dense Branch** MVS textured 3D mesh.
   - Use the **Display panel** (bottom-right of the 3D Viewer) to toggle visibility (Eye icon) between `Texturing_Fast`, `Texturing_Dense`, and `GenSfM_1` for instant side-by-side quality and wireframe comparison.

---

## License

Released under the **Mozilla Public License 2.0 (MPL-2.0)** in compliance with AliceVision Meshroom upstream guidelines.
