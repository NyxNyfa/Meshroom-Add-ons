# Product Requirements Document (PRD): Node `GenSfM` untuk Rekonstruksi 3D dari 2–4 Foto Acak

**Nama Produk / Node:** `GenSfM` (Sparse Random-View AI Initialization & 3D Reconstruction Node)  
**Target Repositori Kontribusi:** [AliceVision Meshroom](https://github.com/alicevision/Meshroom) & [MeshroomHub](https://github.com/meshroomHub/mrHelloWorld)  
**Target Sistem Operasi:** Lintas OS (Windows & Linux, WSL2 Ubuntu dengan NVIDIA CUDA 12.8) pada [Meshroom v2025.1.0+](https://github.com/alicevision/Meshroom/discussions/2839)  
**Mesin AI Inti:** [GenSfM — Generative Structure from Motion with Native 3D Diffusion](https://dotb1ue.github.io/GenSfm/) ([Repositori GitHub dotb1ue/GenSfm](https://github.com/dotb1ue/GenSfm), [Makalah arXiv:2610.35522](https://arxiv.org/html/2610.35522v1))  
**Lisensi Kode:** MPL-2.0 (standar wajib kontribusi [Meshroom](https://github.com/alicevision/Meshroom)) & MIT License ([dotb1ue/GenSfm](https://github.com/dotb1ue/GenSfm))  
**Status Dokumen:** Implementasi Selesai & Terverifikasi (v4.0 Final Baseline)

---

## 1. Latar Belakang Masalah & Visi Produk

### 1.1 Masalah Utama pada Input Gambar di AliceVision Meshroom Saat Ini
Pada alur kerja standar [AliceVision Meshroom](https://github.com/alicevision/Meshroom), node awal `CameraInit` menuntut pengguna memasukkan **30 hingga ratusan foto** yang memutari objek secara rapat dengan area tumpang tindih (*overlap*) antar-foto minimal 60%–80%. Jika pengguna hanya memasukkan **2 hingga 4 foto dari sudut pandang acak** (*sparse, uncalibrated views*):
1. Node `FeatureExtraction` dan `FeatureMatching` gagal menemukan pasangan titik kunci (*keypoints*) yang sama karena sudut antar-foto terlalu lebar (misalnya $90^\circ$–$180^\circ$).
2. Node `StructureFromMotion` menolak foto-foto tersebut (0 kamera terekonstruksi) karena tidak memiliki informasi kalibrasi lensa (*intrinsics*) maupun titik korespondensi yang cukup.
3. Node `DepthMap` dan `Meshing` gagal membentuk permukaan 3D karena bagian objek yang tidak terambil dari banyak sudut menghasilkan lubang besar (*occlusion*).

### 1.2 Solusi: 1 Node `GenSfM` untuk Mengurangi Kebutuhan Init Gambar Menjadi 2–4 Foto Acak
Proyek ini menambahkan **1 node modular bernama `GenSfM`** ke dalam [Meshroom](https://github.com/alicevision/Meshroom) yang memanfaatkan model difusi 3D [GenSfM](https://dotb1ue.github.io/GenSfm/) ([arXiv:2610.35522](https://arxiv.org/html/2610.35522v1)). Alih-alih memotret puluhan kali, pengguna cukup memasukkan **2 hingga 4 foto objek dari sudut acak tanpa kalibrasi** ke `CameraInit`, lalu menghubungkannya ke node `GenSfM`.

Di dalam satu node tersebut, AI [GenSfM](https://github.com/dotb1ue/GenSfm) secara serentak menebak posisi kamera dari tiap foto acak (akurasi $>96\%$ $\text{Acc}@5^\circ$ tanpa kalibrasi awal menurut [arXiv:2610.35522](https://arxiv.org/html/2610.35522v1)) sekaligus merekonstruksi geometri 3D utuh (`mesh.glb` / `mesh.obj`) dan *3D Gaussian Splatting* (`gaussian.ply`), serta menerjemahkan hasilnya kembali ke format standar `SfMData` (`.abc` / `.sfm`) milik [AliceVision](https://alicevision.org/).

### 1.3 Perbandingan Kebutuhan Input: Meshroom Standar vs. Meshroom + Node `GenSfM`

| Parameter Input & Proses | Pipeline [Meshroom](https://github.com/alicevision/Meshroom) Standar | Pipeline [Meshroom](https://github.com/alicevision/Meshroom) dengan Node `GenSfM` |
| :--- | :--- | :--- |
| **Jumlah Foto di `CameraInit`** | 30 – 150+ foto berurutan dari seluruh sudut | **Cukup 2 – 4 foto** dari sudut pandang acak yang berbeda ([dotb1ue/GenSfm](https://github.com/dotb1ue/GenSfm)) |
| **Syarat Tumpang Tindih (*Overlap*)** | Wajib tinggi (>60% area objek harus sama antar-foto berdekatan) | **Bebas / Acak** (cukup menampilkan sisi-sisi objek kaku yang sama) |
| **Kalibrasi Kamera Awal (EXIF/SensorDB)** | Sangat bergantung pada database sensor kamera [AliceVision](https://github.com/alicevision/AliceVision) | **Tidak butuh kalibrasi awal** (*uncalibrated views*, intrinsik diestimasi otomatis oleh [GenSfM](https://arxiv.org/html/2610.35522v1)) |
| **Penanganan Sisi Objek yang Terhalang** | Berlubang / gagal di-mesh jika kurang foto | Diisi secara utuh menggunakan *3D object-level priors* dari difusi 3D berbasis [TRELLIS](https://github.com/dotb1ue/GenSfm) |
| **Beban Komputasi di Laptop Pengguna** | Berat di CPU & GPU lokal selama puluhan menit | **Ringan di Windows/Laptop** dengan GPU lokal via WSL2 atau offload ke **Cloud AI API** |

---

## 2. Arsitektur Integrasi di Dalam Graph Editor Meshroom

Node `GenSfM` dirancang agar **tidak merusak atau menghapus node bawaan** [Meshroom](https://github.com/alicevision/Meshroom), melainkan menjadi jembatan pintar tepat setelah `CameraInit`. Node ini menyediakan dua opsi integrasi utama:

### 2.1 Jalur 1: Pipeline Mandiri `GenSfM.mg` (All-in-One 3-Node)
Pengguna cukup memasukkan 2–4 foto ke `CameraInit` dan menyambungkannya ke `GenSfM`. Node secara otomatis memotong background menggunakan modul AI `u2net` bawaan, merekonstruksi 3D mesh dan 3D Gaussian Splats, lalu meneruskannya ke `Publish`:
```text
[CameraInit (2–4 Foto Acak)] ──► [Node GenSfM (auto_rembg)] ──► [Publish_Outputs]
```

### 2.2 Jalur 2: Integrasi Modular Bersama `mrRemBG` (`GenSfM_RemBG.mg`)
Untuk objek dengan kontur rumit, helai rambut, bulu, atau dedaunan, segmentasi didelegasikan secara bersih ke node `RemBG`:
```text
[CameraInit] ──► [RemBG (U2-Net / ISNet)] ──┬── (outputSfm) ──► [GenSfM (external_masks)] ──► [Publish_Outputs]
                                            └── (outputMasks) ─┘
```
- **Koneksi Node:** Port `RemBG.outputMasks` dihubungkan langsung ke atribut `GenSfM.inputMasks`, dengan parameter `maskingStrategy` diatur ke `"external_masks"`.
- **Keuntungan:** Memisahkan beban pra-pemrosesan citra dari difusi 3D dan memungkinkan inspeksi visual kualitas mask di 2D Viewer sebelum memicu inferensi model difusi.

---

## 3. Spesifikasi Fungsional & Atribut Node `GenSfM.py`

### 3.1 Metadata Kelas Node

| Properti Kelas | Nilai | Keterangan Standar [Meshroom](https://github.com/alicevision/Meshroom) |
| :--- | :--- | :--- |
| `__version__` | `"1.0"` | Wajib untuk melacak perubahan versi dan memicu *cache invalidation* otomatis. |
| `category` | `"Sparse Reconstruction / AI"` | Sub-kategori di menu klik-kanan Graph Editor. |
| `gpu` | `desc.Level.INTENSIVE` | Penanda kebutuhan akselerasi GPU NVIDIA. |
| `size` | `desc.DynamicNodeSize("input")` | Mengikuti jumlah *views* pada node input (`CameraInit`). |

### 3.2 Spesifikasi Atribut Input (`inputs`)

| Nama Atribut | Tipe Descriptor | Default | Invalidate? | Deskripsi Teknis & Aturan Validasi |
| :--- | :--- | :--- | :--- | :--- |
| `input` | `desc.File` | `""` | `True` | Path ke file `sfmData` (`.sfm` / `.abc`) keluaran `CameraInit` atau direktori berisi 2–4 foto acak. |
| `inputMasks` | `desc.File` | `""` | `True` | *(Opsional)* Path ke folder mask RGBA/grayscale dari node `RemBG`. |
| `maskingStrategy` | `desc.ChoiceParam` | `"auto_rembg"` | `True` | Pilihan: `"auto_rembg"` (jalankan `rembg` bawaan GenSfM), `"existing_alpha"` (kanal alpha PNG input), atau `"external_masks"` (gunakan `inputMasks`). |
| `viewSelectionMode` | `desc.ChoiceParam` | `"all_up_to_4"` | `True` | Cara memilih foto dari `CameraInit`: `"all_up_to_4"`, `"manual_indices"`, atau `"auto_diverse"`. |
| `selectedIndices` | `desc.StringParam` | `"0,1,2,3"` | `True` | Daftar indeks foto apabila `viewSelectionMode == "manual_indices"`. |
| `executionBackend` | `desc.ChoiceParam` | `"local_wsl_gpu"` | `False` | Backend eksekusi: `"local_wsl_gpu"`, `"local_docker"`, atau `"cloud_api"`. |
| `seed` | `desc.IntParam` | `1` | `True` | *Random seed* untuk proses *denoising* difusi (`0` – `2147483647`). |
| `steps` | `desc.IntParam` | `25` | `True` | Jumlah langkah *denoising* difusi Stage 1 & Stage 2 (`10` – `100`). |
| `lowVramMode` | `desc.BoolParam` | `True` | `False` | Mengaktifkan strategi sequential stage unloading untuk GPU laptop 4GB. |
| `generateSyntheticViews` | `desc.IntParam` | `0` | `True` | Jumlah gambar sudut pandang sintetis orbital tambahan (`0` = nonaktif). |
| `cloudEndpoint` | `desc.StringParam` | `""` | `False` | URL server inferensi Cloud AI jika menggunakan mode remote. |
| `cloudApiKey` | `desc.StringParam` | `""` | `False` | Kunci autentikasi API Cloud. |

### 3.3 Spesifikasi Atribut Output (`outputs`)

| Nama Atribut | Tipe Descriptor | Nilai Path Output | Fungsi & Kompatibilitas |
| :--- | :--- | :--- | :--- |
| `outputSfMData` | `desc.File` | `"{nodeCacheFolder}/sfmData.abc"` | Berisi daftar gambar, intrinsik kamera, pose ekstrinsik hasil prediksi, dan *sparse point cloud* dalam format AliceVision. |
| `outputMeshObj` | `desc.File` (`3d`) | `"{nodeCacheFolder}/mesh.obj"` | Mesh 3D bertekstur (`.obj` + `.mtl` + `texture.png`) yang langsung tampil di 3D Viewport. |
| `outputMeshGlb` | `desc.File` | `"{nodeCacheFolder}/mesh.glb"` | File *Textured Mesh* asli keluaran GenSfM dalam koordinat Y-up `(x, z, -y)` untuk ekspor ke Blender/Unity/Unreal. |
| `outputGaussianPly` | `desc.File` (`3d`) | `"{nodeCacheFolder}/gaussian.ply"` | File *3D Gaussian Splatting* asli keluaran GenSfM. |
| `outputCamerasJson` | `desc.File` | `"{nodeCacheFolder}/cameras.json"` | File JSON ekstrinsik OpenCV Z-up, intrinsik ternormalisasi, dan matriks `data_to_mesh`. |
| `outputMaskedInputs` | `desc.File` | `"{nodeCacheFolder}/input-rgba-*.png"` | Gambar objek berlatar transparan (RGBA) untuk inspeksi di 2D Viewer. |
| `outputReprojections` | `desc.File` | `"{nodeCacheFolder}/reprojection-*.png"` | Render mesh bertekstur dari sudut pandang kamera terprediksi untuk memvalidasi akurasi pose. |

---

## 4. Spesifikasi Teknis Algoritma Internal

### 4.1 Tahap 1: Ekstraksi Gambar & Masking RGBA
1. **Pembacaan `CameraInit`:** Node membaca daftar `views` dari `.sfm` JSON `CameraInit`. Validasi minimal 2 foto dan optimal 4 foto.
2. **Penanganan Masking:**
   - Mode `auto_rembg`: mengekstrak siluet objek dengan U2-Net internal.
   - Mode `external_masks`: menyatukan foto RGB dengan kanal alpha dari mask `mrRemBG`.

### 4.2 Tahap 2: Proses Inferensi 2-Stage GenSfM
1. **DINOv2 Encoder:** Mengekstrak representasi fitur padat tanpa informasi lensa/pose awal.
2. **Stage 1 (Shape & Camera Pose Generation):** Joint Diffusion Transformer (DiT) berbasis 3D-native UV-volume memprediksi geometri kasar dan pose kamera ekstrinsik/intrinsik dari foto acak.
3. **Stage 2 (Pose-Aware Fine Geometry & Texture):** Menggunakan pose Stage 1 untuk memproyeksikan fitur semantik pada *active voxel* dan menghasilkan `mesh.glb`, `gaussian.ply`, serta `cameras.json`.

### 4.3 Tahap 3: Konversi Koordinat ke AliceVision SfMData
- **Transformasi Ekstrinsik:** Matriks *world-to-camera* $\mathbf{T}_{w2c}$ dikalikan invers dari `data_to_mesh` ($\mathbf{T}_{\text{mesh\_w2c}} = \mathbf{T}_{w2c} \cdot \mathbf{M}_{\text{d2m}}^{-1}$) sehingga kamera berada pada kerangka koordinat yang sama persis dengan mesh 3D.
- **Denormalisasi Intrinsik:** Mengalikan panjang fokus ternormalisasi ($f_x, f_y$) dan principal point ($c_x, c_y$) dengan resolusi piksel asli $(W, H)$.
- **Injeksi Landmarks:** Memasukkan sampel titik 3D dan warna RGB ke dalam `sfmData.abc` agar langsung tampil di 3D Viewport Meshroom.

---

## 5. Rencana Pengujian & Kriteria Keberhasilan

- [x] Node `GenSfM` terdeteksi di Meshroom 2025.1.0 pada Windows via plugin junction.
- [x] Eksekusi inferensi sukses via WSL2 Ubuntu (CUDA 12.8, PyTorch 2.9.1, GPU RTX 3050 Laptop 4GB).
- [x] Template resmi `GenSfM.mg` dan `GenSfM_RemBG.mg` berjalan mulus dari GUI Meshroom.
- [x] Menghasilkan mesh 360° watertight, file glTF PBR, dan 3D Gaussian Splats (.ply) dari 2–4 foto acak.
