# Product Requirements Document (PRD): Meshroom Single-Image AI 3D Reconstruction & Gaussian Splatting Add-on (`mrAIGen3D`)

**Nama Proyek:** `mrAIGen3D` (Meshroom Generative 3D Node Add-on)  
**Target Aplikasi Host:** [AliceVision Meshroom 2025.1.0+](https://github.com/alicevision/meshroom) (Lisensi MPLv2)  
**Arsitektur Eksekusi:** Hibrida — Subproses Lokal WSL2 (NVIDIA GeForce RTX 3050 Laptop 4 GB VRAM / CPU) & Remote GPU Worker via HTTP Tunnel (Google Colab Tesla T4 16 GB / L4 / A100)  
**Tanggal Dokumen:** 12 November 2026  
**Status:** Implementasi Selesai & Terverifikasi (Baseline v1.0)

---

## 1. Ringkasan Eksekutif & Tujuan Produk

Proyek `mrAIGen3D` menghadirkan *custom node* **`AI3DGenerator`** (Kategori: `AI Generation / 3D Reconstruction`) di dalam ekosistem visual [AliceVision Meshroom](https://github.com/alicevision/meshroom). Node ini mengubah **satu gambar 2D tunggal** menjadi **aset mesh 3D bertekstur (`.obj` dan `.glb`)** atau **3D Gaussian Splatting point cloud (`.ply`)** menggunakan model *state-of-the-art* (SOTA):

1. **Model Inferensi Lokal (Khusus Hardware Laptop RTX 3050 4 GB VRAM):**
   - **[stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR):** Rekonstruksi mesh eksplisit cepat (<4 detik) berbasis NeRF transformer dan Marching Cubes beresolusi teroptimasi ($192^3$).
   - **[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat.git):** Rekonstruksi feed-forward 3D Gaussian Splatting berkecepatan tinggi yang menghasilkan 131.072 titik Gaussian dalam format `.ply`.
2. **Model Remote Cloud (Google Colab GPU Worker via HTTP Tunnel):**
   - **[stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)** (SF3D)
   - **[stabilityai/stable-point-aware-3d](https://huggingface.co/stabilityai/stable-point-aware-3d)** (SPAR3D)
   - **[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS)**
   - **[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/trellis.2)**
   - **[tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)** (2-stage Sequential DiT + Paint)

---

## 2. Keputusan Desain Kunci (Architectural Decisions)

| Kode | Area Arsitektur | Keputusan Desain yang Dikunci | Rasionalitas Teknis & Referensi |
| :--- | :--- | :--- | :--- |
| **D1** | **Pemisahan Modular (`RemBG` $\rightarrow$ `AI3DGenerator`)** | Menjadikan penghapusan background sebagai node terpisah (`RemBG`), sehingga `AI3DGenerator` berfokus murni pada inferensi rekonstruksi 3D. | Memaksimalkan fleksibilitas: pengguna dapat meninjau hasil pemotongan latar belakang di 2D Viewer sebelum masuk ke inferensi 3D. |
| **D2** | **Optimasi Memori Laptop 4GB VRAM** | Menerapkan `chunk_size = 4096`, `mc_resolution = 192`, FP16, compositing RGBA netral gray, dan pembersihan VRAM eksplisit (`torch.cuda.empty_cache()`). | Menjamin TripoSR berjalan bebas dari *CUDA Out-of-Memory (OOM)* pada GPU laptop kelas konsumen (RTX 3050 4GB). |
| **D3** | **Dukungan 3D Gaussian Splatting (`TripoSplat`)** | Mengintegrasikan model feed-forward 3DGS dari VAST-AI Research ke dalam pilihan `modelName` dengan output format biner `.ply`. | Memungkinkan Meshroom merender Gaussian Splats densitas tinggi secara instan dari 1 gambar. |
| **D4** | **Jembatan Remote Colab (HTTP REST Tunnel)** | Menggunakan server `FastAPI` + `Cloudflared` pada Google Colab dengan endpoint `/health` dan `/generate`. | Memungkinkan pengguna mengeksekusi model raksasa (Hunyuan3D-2.1 29GB / TRELLIS 16GB) di cloud tanpa membebani GPU laptop. |
| **D5** | **Pipeline Terpadu 1-Klik (`AIGen3D.mg`)** | Menyediakan template pipeline terpadu `AIGen3D.mg` yang langsung mengintegrasikan `CameraInit` ➔ `RemBG` ➔ `AI3DGenerator` ➔ `Publish`. | Memastikan alur 1-foto menghasilkan mesh 3D bersih tanpa background box dalam ~4 detik, dengan opsi penambahan `MeshFiltering` manual jika dibutuhkan. |

---

## 3. Spesifikasi Teknis Parameter Node `AI3DGenerator`

### 3.1 Parameter Input (`inputs`)
- `input` (`desc.File`): Single unified input. Menerima koneksi dari `RemBG.outputSfm`, `CameraInit.output` (`cameraInit.sfm`), direktori folder gambar, atau file gambar langsung (`.png`/`.jpg`).
- `selectedViewIndex` (`desc.IntParam`, default: `0`): Indeks sudut pandang gambar yang dipilih jika input memuat beberapa foto.
- `modelName` (`desc.ChoiceParam`, default: `"triposr"`): `["triposr", "triposplat", "stable-fast-3d", "spar3d", "trellis-large", "trellis-2-4b", "hunyuan3d-2.1"]`.
- `executionBackend` (`desc.ChoiceParam`, default: `"local_wsl_gpu"`): `["local_wsl_gpu", "local_cpu", "colab_remote"]`.
- `remoteEndpointUrl` (`desc.StringParam`, default: `""`): URL publik dari notebook Google Colab.
- `mcResolution` (`desc.ChoiceParam`, default: `"192"`): `["192", "256", "384"]`.
- `chunkSize` (`desc.ChoiceParam`, default: `"4096"`): `["2048", "4096", "8192"]`.
- `numGaussians` (`desc.ChoiceParam`, default: `"131072"`): `["32768", "65536", "131072", "262144"]`.
- `textureResolution` (`desc.ChoiceParam`, default: `"1024"`): `["512", "1024", "2048"]`.
- `lowVramMode` (`desc.BoolParam`, default: `True`): Mode pemuatan memori hemat VRAM.
- `hunyuanSkipPaint` (`desc.BoolParam`, default: `False`): Opsi skip paint khusus Hunyuan3D-2.1.
- `hfToken` (`desc.StringParam`, default: `""`): Token akses Hugging Face Hub.

### 3.2 Parameter Output (`outputs`)
- `outputMeshObj` (`desc.File`, `semantic="3d"`): `{nodeCacheFolder}/mesh.obj` — **Langsung dirender di 3D Viewport Meshroom**.
- `outputMeshGlb` (`desc.File`): `{nodeCacheFolder}/asset_pbr.glb` — Format glTF 2.0 PBR komplit.
- `outputGaussianPly` (`desc.File`, `semantic="3d"`): `{nodeCacheFolder}/gaussian.ply` — File 3D Gaussian Splats.
- `outputPreviewPng` (`desc.File`, `semantic="image"`): `{nodeCacheFolder}/preview.png` — Sinkronisasi foto di 2D Viewer.

---

## 4. Spesifikasi Pipeline Template

**`AIGen3D.mg` (Generasi 3D Instan 1 Foto):**
```text
CameraInit -> RemBG -> AI3DGenerator -> Publish_Outputs
```

---

## 5. Kepatuhan Lisensi & Atribusi Legal

- Wrapper plugin berlisensi **MPLv2** (sama dengan lisensi AliceVision Meshroom).
- **Klausul Wajib Stability AI Community License Agreement (5 Juli 2024):**
  - Menyertakan berkas `Notice` verbatim di root repositori:
    > `This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved`
  - Menyertakan salinan lengkap `LICENSE_STABILITY_AI.md`.
  - Menampilkan teks *"Powered by Stability AI"* pada docstring node `AI3DGenerator` dan README.
- Model `TripoSplat` tunduk pada lisensi **MIT** (VAST-AI Research & Tripo AI).

---

## 6. Kriteria Penerimaan & Verifikasi

- [x] Node `AI3DGenerator` terdaftar di kategori `AI Generation / 3D Reconstruction`.
- [x] Eksekusi model lokal `TripoSR` selesai dalam <5 detik pada GPU RTX 3050 Laptop (4GB VRAM) tanpa OOM.
- [x] Hasil mesh `.obj` langsung muncul secara otomatis di panel 3D Viewport Meshroom saat di-double-click.
- [x] Lolos seluruh test suite (`test_ai3d.py`) dengan status 100% PASS.
