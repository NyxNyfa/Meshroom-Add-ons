# Product Requirements Document (PRD): Meshroom AI Background Removal & Alpha Matting Add-on (`mrRemBG`)

**Nama Proyek:** `mrRemBG` (Meshroom Foreground Matting & Alpha Extraction Plugin)  
**Target Aplikasi Host:** [AliceVision Meshroom 2025.1.0+](https://github.com/alicevision/meshroom) (MPLv2)  
**Arsitektur Eksekusi:** Subproses Lokal Hibrida — WSL2 Ubuntu GPU (NVIDIA CUDA 12.8) & Windows Python Native Fallback  
**Tanggal Dokumen:** 12 November 2026  
**Status:** Implementasi Selesai & Terverifikasi (Baseline v1.0)

---

## 1. Ringkasan Eksekutif & Tujuan Produk

Dalam alur kerja rekonstruksi 3D fotogrametri konvensional maupun generatif AI (seperti `mrAIGen3D` dan `mrGenSfM`), piksel latar belakang (*background clutter*) yang tidak relevan sering kali menimbulkan artefak geometri berupa *bounding box*, serpihan poligon melayang, atau kegagalan pencocokan titik fitur kamera.

Proyek `mrRemBG` bertujuan membangun *custom node* resmi di dalam ekosistem [AliceVision Meshroom](https://github.com/alicevision/meshroom) bernama **`RemBG`** (Kategori: `AI Generation / Segmentation`). Node ini bertugas:
1. Menghapus latar belakang secara otomatis (*zero-click foreground extraction*) menggunakan arsitektur deep learning SOTA (`U2-Net`, `U2-Net-p`, `ISNet`, dan `BiRefNet`).
2. Membaca metadata kamera dari node `CameraInit` (`cameraInit.sfm`) dan mengekspor kembali file `sfmData.sfm` baru yang mengarahkan path gambar ke format PNG transparan (RGBA) tanpa merusak metadata sensor kamera, intrinsik, maupun orientasi pose.
3. Menyediakan mask biner grayscale terpisah untuk dihubungkan langsung ke slot `inputMasks` pada node `GenSfM`.

---

## 2. Keputusan Desain Kunci (Architectural Decisions)

| Kode | Area Desain | Keputusan Desain yang Dikunci | Rasionalitas Teknis & Dampak |
| :--- | :--- | :--- | :--- |
| **D1** | **Pemisahan Modul (Decoupled Node)** | Memisahkan fungsionalitas background removal ke plugin independen `mrRemBG`, alih-alih menyatukannya secara permanen di dalam node 3D. | Memungkinkan node digunakan secara serbaguna lintas pipeline: untuk `AI3DGenerator`, untuk `GenSfM`, maupun untuk pipeline fotogrametri standar AliceVision. |
| **D2** | **Preservasi SfMData & Normalisasi Path** | Membaca struktur JSON `sfmData` dan mengonversi path biner dari WSL (`/mnt/d/...`) ke path Windows (`D:/...`). | Menjamin GUI Meshroom di OS Windows dapat membaca dan merender gambar transparan tanpa galat format path Linux. |
| **D3** | **Dual-Engine Execution (WSL2 GPU + Windows)** | Menjalankan inferensi matting utama di dalam WSL2 Ubuntu menggunakan environment conda `gensfm` yang telah terpasang CUDA 12.8 dan PyTorch. | Menghindari kompilasi dependensi C++/ONNX yang rumit di Windows dan memaksimalkan kecepatan inferensi GPU RTX 3050 (<4 detik per gambar). |
| **D4** | **Fitur Pad to Square (1:1)** | Menyediakan parameter `padToSquare` untuk menambahkan padding transparan simetris ke rasio aspek 1:1. | Mencegah distorsi peregangan rasio aspek pada jaringan saraf 3D generatif (`TripoSR`, `TripoSplat`, `GenSfM`). |

---

## 3. Spesifikasi Fungsional Node `RemBG`

### 3.1 Input Attributes (`inputs`)
- `input` (`desc.File`): Single unified input. Menerima koneksi dari `CameraInit.output` (`cameraInit.sfm`), direktori folder gambar, atau file gambar tunggal (`.png`, `.jpg`, `.webp`, `.bmp`).
- `model` (`desc.ChoiceParam`, default: `"u2net"`): Pilihan model neural network (`u2net`, `u2netp`, `isnet-general-use`, `birefnet-general`).
- `alphaMatting` (`desc.BoolParam`, default: `False`): Mengaktifkan estimasi trimap dan matting morfologi untuk helai rambut dan bulu halus.
- `alphaMattingForegroundThreshold` (`desc.IntParam`, default: `240`): Ambang piksel foreground pasti.
- `alphaMattingBackgroundThreshold` (`desc.IntParam`, default: `10`): Ambang piksel background pasti.
- `alphaMattingErodeSize` (`desc.IntParam`, default: `10`): Ukuran kernel erosi pembentukan trimap.
- `padToSquare` (`desc.BoolParam`, default: `False`): Padding transparan ke rasio 1:1.
- `executionBackend` (`desc.ChoiceParam`, default: `"local_wsl_gpu"`): Pilihan runtime (`local_wsl_gpu`, `local_windows`, `local_cpu`).

### 3.2 Output Attributes (`outputs`)
- `outputSfm` (`desc.File`): Path ke `{nodeCacheFolder}/sfmData.sfm` (diperbarui dengan view RGBA).
- `outputFolder` (`desc.File`): Direktori `{nodeCacheFolder}/rgba_images` berisi file PNG transparan.
- `outputMasks` (`desc.File`): Direktori `{nodeCacheFolder}/masks` berisi mask grayscale (1 channel).
- `outputPreview` (`desc.File`, `semantic="image"`): Sinkronisasi visual dengan panel **2D Viewer** Meshroom (`preview_<VIEW_ID>.png`).

---

## 4. Topologi Pipeline Bawaan

1. **`RemBG.mg` (Standar Segmentasi):**
   ```text
   CameraInit -> RemBG -> Publish_Outputs
   ```
2. **`GenSfM_RemBG.mg` (Integrasi Fotogrametri AI):**
   ```text
   CameraInit -> RemBG -> GenSfM (inputMasks = RemBG.outputMasks, maskingStrategy = 'external_masks') -> Publish_Outputs
   ```

---

## 5. Kriteria Penerimaan & Verifikasi (Acceptance Criteria)

- [x] Node `RemBG` muncul di kategori `AI Generation / Segmentation` di palette Meshroom.
- [x] Mampu membaca `cameraInit.sfm` multi-gambar dan menuliskan kembali `sfmData.sfm` valid.
- [x] Output gambar PNG berkanal 4 (RGBA) transparan dengan padding 1:1 opsional.
- [x] Lolos pengujian unit test terotomatisasi (`test_rembg.py`) dengan status 100% PASS.
