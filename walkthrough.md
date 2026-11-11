# Walkthrough: Penyelesaian Out-of-Memory (OOM) & Eksekusi GenSfM pada NVIDIA RTX 3050 Laptop GPU (4GB VRAM)

Dokumen ini mendokumentasikan secara rinci analisis teknis, akar masalah, arsitektur perbaikan, dan panduan langkah demi langkah penggunaan modul `mrGenSfM` pada GPU laptop dengan VRAM terbatas (4.0 GB) melalui lingkungan WSL2 Ubuntu di AliceVision Meshroom.

---

## 1. Latar Belakang & Analisis Permasalahan

Saat menjalankan proses rekonstruksi 3D berbasis AI native GenSfM menggunakan backend `local_wsl_gpu` pada GPU laptop:
* **GPU Terdeteksi:**
  * GPU 0: Intel(R) Iris(R) Xe Graphics (Integrated, shared RAM sistem).
  * GPU 1: NVIDIA GeForce RTX 3050 Laptop GPU (Dedicated VRAM 4.0 GB).
* **Gejala Masalah:**
  1. Penggunaan GPU 1 (RTX 3050) melonjak naik, lalu eksekusi terhenti dengan galat:
     ```text
     RuntimeError: CUDA driver error: out of memory
     ```
  2. Ketika proses dieksekusi di dalam WSL2 Ubuntu, sistem sempat mengalami penghentian paksa (*crash*) dengan kode:
     ```text
     Out of memory: Killed process (python3) ... anon-rss: 7479276kB
     Wsl/Service/E_UNEXPECTED
     ```

### Klarifikasi Terkait Intel Iris Xe
Intel Iris Xe Graphics **bukan** penyebab dari *out of memory*. Berdasarkan Task Manager dan log PyTorch:
* PyTorch dan modul CUDA secara tepat memilih **GPU 1 (NVIDIA GeForce RTX 3050)**.
* Lonjakan grafik VRAM terjadi secara langsung pada dedicated memory GPU 1.
* Masalah utama berakar pada kebutuhan bawaan (*default*) arsitektur Trellis/GenSfM yang dirancang untuk GPU workstation (24 GB VRAM) dan RAM sistem besar (32–64 GB).

---

## 2. Analisis Akar Masalah (Root Causes)

Secara bawaan, eksekusi resmi GenSfM (`example_multi_image.py` / `trellis` evaluator) memerlukan memori yang melampaui batas GPU 4.0 GB karena faktor-faktor berikut:

| No | Komponen Penyebab OOM | Deskripsi Beban Memori | Akibat pada RTX 3050 (4GB) |
| :--- | :--- | :--- | :--- |
| **1** | **Inisialisasi Monolitik Evaluator** | Memuat 6 model sekaligus saat `__init__` (`stage1_joint`, `stage2_slat`, `dinov2`, `ss_decoder`, `uv_decoder`, `slat_decoder_gs`, `slat_decoder_mesh`). | VRAM langsung terpakai **> 4.2 GB** sebelum difusi dimulai. |
| **2** | **Dense Grid FlexiCubes ($256^3$)** | `SparseFeatures2Mesh` membangun grid statis (`reg_c` int64 dan `reg_v` float32) serta tensor atribut bervolume dense $257^3$. | Alokasi mendadak sebesar **~2.5 GB VRAM**. |
| **3** | **Multiview Texture Rasterization** | Fungsi `to_glb` merender 100 sudut pandang kamera berukuran $1024 \times 1024$ dengan mode optimasi gradien bertumpuk (`diff_attrs='all'`). | Pustaka `nvdiffrast` kehabisan VRAM saat proses teksturing. |
| **4** | **Batas Memori Virtual Bawaan WSL2** | Microsoft menetapkan batas RAM default WSL2 sebesar 50% RAM Windows (7.6 GB) dengan swap kecil (2 GB). | Linux kernel OOM Killer menghentikan proses Python saat membaca bobot safetensors. |

---

## 3. Arsitektur Solusi & Optimalisasi yang Diterapkan

Untuk mengatasi keterbatasan tersebut tanpa menurunkan kualitas output rekonstruksi 3D, diterapkan 4 pilar optimasi:

```text
[Input 4 Views]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. STAGE 1: Sparse Structure & Camera Poses                 │
│    • Muat HANYA model Stage 1 (denoiser, ss/uv decoder)     │
│    • Muat DINOv2 ViT-L untuk conditioning gambar            │
│    • Peak VRAM: ~2.62 GB (Aman di bawah 4.0 GB)             │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Pose camera terselesaikan)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Transisi & Pembersihan Memori #1                            │
│    • Hapus model Stage 1 & DINOv2 dari RAM dan VRAM         │
│    • VRAM turun drastis ke ~93 MB                           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. STAGE 2: SLat Appearance & Fine Geometry                 │
│    • Muat HANYA SLat Denoiser (2.3 GB) secara independen    │
│    • Jalankan 25-step diffusion sampling                    │
│    • Peak VRAM: ~1.24 GB                                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Latent tensor x_0 selesai)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Transisi & Pembersihan Memori #2                            │
│    • Hapus SLat Denoiser dari VRAM & RAM                    │
│    • VRAM turun ke ~99 MB                                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DECODING & EXTRACTION: CPU FlexiCubes                    │
│    • Lapisan saraf Mesh Decoder diproses di GPU             │
│    • Ekstraksi FlexiCubes dense grid dialihkan ke CPU RAM   │
│    • Mengeliminasi lonjakan 2.5 GB VRAM                     │
│    • Muat & decode 3D Gaussian Splats secara sekuensial     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. FAST STREAMING TEXTURE BAKING                            │
│    • Mode baking tekstur diubah ke streaming per-view       │
│    • Tidak menumpuk 100 buffer rasterisasi di VRAM          │
│    • Ekspor: mesh.glb, gaussian.ply, cameras.json           │
└─────────────────────────────────────────────────────────────┘
```

### Rincian Perubahan Kode

1. **Konfigurasi Memori WSL2 (`C:\Users\Nyfa\.wslconfig`)**
   Dibuat file konfigurasi host untuk memperluas memori virtual WSL2:
   ```ini
   [wsl2]
   memory=10GB
   swap=16GB
   ```
   *Total ruang memori virtual WSL2 meningkat menjadi **25.7 GB**, mencegah Linux OOM Killer.*

2. **Sequential Lazy Loading (`LOW_VRAM_WRAPPER_SCRIPT`)**
   File: `D:\Meshroom\mrGenSfM\gensfm_core\wsl_docker_bridge.py`
   * Model diinisialisasi secara bertahap (*on-demand*).
   * Stage 1 models langsung di-`del` dan dibersihkan via `gc.collect()` serta `torch.cuda.empty_cache()` sebelum Stage 2 dimuat.
   * `evaluator.models["slat_denoiser"]` dimuat mandiri, dijalankan, lalu langsung dibersihkan sebelum decoder dimuat.

3. **CPU-Offloaded FlexiCubes Mesh Extraction**
   Ekstraksi kubus dense dialihkan ke CPU:
   ```python
   decoder_mesh.mesh_extractor = c2m.SparseFeatures2Mesh(
       device="cpu",
       res=decoder_mesh.resolution * 4,
       use_color=decoder_mesh.rep_config.get("use_color", False),
   )
   def cpu_to_representation(x):
       return [decoder_mesh.mesh_extractor(x[i].cpu(), training=decoder_mesh.training) for i in range(x.shape[0])]
   decoder_mesh.to_representation = cpu_to_representation
   ```

4. **Fast-Stream Texture Baking**
   Mencegah penumpukan 100 frame di `nvdiffrast`:
   ```python
   orig_to_glb = pu.to_glb
   def fast_to_glb(*g_args, **g_kwargs):
       orig_bake = pu.bake_texture
       pu.bake_texture = lambda *b_args, **b_kwargs: orig_bake(*b_args, **dict(b_kwargs, mode="fast"))
       try:
           return orig_to_glb(*g_args, **g_kwargs)
       finally:
           pu.bake_texture = orig_bake
   pu.to_glb = fast_to_glb
   ```

---

## 4. Hasil Uji & Verifikasi Sistem

Pengujian dilakukan secara bertingkat:

### A. Uji Unit Test Plugin
Dijalankan pada environment Windows host:
```powershell
python -m unittest discover -s D:\Meshroom\mrGenSfM\tests
```
**Hasil:**
```text
Ran 9 tests in 3.506s
OK
```

### B. Uji End-to-End Bridge (Windows Host $\rightarrow$ WSL2 RTX 3050)
Dijalankan menggunakan 4 foto wagon uncalibrated:
```text
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Initializing sequential Stage 1 execution...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Running Stage 1 (sparse structure & camera poses)...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Solved camera poses for 4 views.
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Computing visual latent conditioning...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Loading Stage 2 (SLat appearance flow model)...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Running Stage 2 (SLat diffusion sampling)...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Decoding 3D Mesh geometry with CPU FlexiCubes...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Decoding 3D Gaussian Splats...
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Exporting GLB, PLY, and camera reprojections...
[WSL2 GenSfM] INFO- Loaded 7655 vertices and 15368 faces.
[WSL2 GenSfM] Rendering: 100it [00:01, 71.88it/s]
[WSL2 GenSfM] [mrGenSfM Low-VRAM] Reconstruction finished successfully
```

### C. File Output yang Dihasilkan
Output berhasil disimpan di folder cache Meshroom:
* `mesh.glb` (1.6 MB) — Model 3D bervolume utuh dan bertekstur.
* `gaussian.ply` (14.8 MB) — Point cloud 3D Gaussian Splats.
* `cameras.json` (4.9 KB) — Matriks kalibrasi intrinsik dan pose ekstrinsik kamera OpenCV format.
* `reprojection-00.png` s/d `reprojection-03.png` — Validasi reproyeksi kamera terhadap objek asli.

---

## 5. Panduan Penggunaan di Meshroom GUI

Ikuti langkah-langkah berikut untuk menjalankan rekonstruksi di Meshroom:

1. **Buka Meshroom:**
   Jalankan Meshroom dan buka proyek:
   * `D:\Meshroom\mrGenSfM_Hybrid_Demo.mg` (atau `mrGenSfM_Wagon_Demo.mg`).

2. **Periksa Node `GenSfM_1`:**
   Klik node **GenSfM_1** di Graph Editor, lalu periksa panel Attributes di sebelah kanan:
   * **executionBackend:** `local_wsl_gpu` *(default)*
   * **lowVramMode:** `True` *(aktifkan mode hemat VRAM)*
   * **generateSyntheticViews:** `24` *(menghasilkan 24 sudut pandang orbital untuk branch photogrammetry)*
   * **removeBackground:** `False` jika citra input sudah berlatar belakang transparan (RGBA), atau `True` jika menggunakan foto berlatar belakang biasa.

3. **Jalankan Rekonstruksi:**
   * Klik kanan pada node **GenSfM_1** $\rightarrow$ pilih **Compute** (atau klik tombol **Recompute** jika node sudah berstatus selesai/merah).
   * Anda dapat memantau log proses di tab **Log** pada panel kanan bawah.

4. **Visualisasi Hasil:**
   * Setelah node berubah warna menjadi hijau:
     * Klik ganda node **GenSfM_1** untuk menampilkan mesh 3D di **3D Viewer**.
     * Untuk melihat hasil downstream branch, jalankan node **Publish_Comparison** atau **Texturing_Fast** untuk melihat UV texture mapping beresolusi 4K.

---

## 6. Ringkasan File Kunci

* [`D:\Meshroom\mrGenSfM\gensfm_core\wsl_docker_bridge.py`](file:///D:/Meshroom/mrGenSfM/gensfm_core/wsl_docker_bridge.py) — Skrip bridge Windows-to-WSL2 dan kode Low-VRAM runner wrapper.
* [`D:\Meshroom\mrGenSfM\meshroom\genSfm\GenSfM.py`](file:///D:/Meshroom/mrGenSfM/meshroom/genSfm/GenSfM.py) — Definisi node Meshroom resmi dan pengaturan parameter GUI.
* [`D:\Meshroom\mrGenSfM\meshroom\genSfm\AiRemoveBackground.py`](file:///D:/Meshroom/mrGenSfM/meshroom/genSfm/AiRemoveBackground.py) — Node dedicated AI Background Removal (AIRMBG).
* [`C:\Users\Nyfa\.wslconfig`](file:///C:/Users/Nyfa/.wslconfig) — Konfigurasi batas RAM dan swap kernel WSL2.
* [`D:\Meshroom\mrGenSfM_Hybrid_Demo.mg`](file:///D:/Meshroom/mrGenSfM_Hybrid_Demo.mg) — File project template siap pakai.

---

## 7. Penanganan Dinding Samping (Studio Backdrop Artifacts) & Integrasi AI Background Removal (AIRMBG)

### A. Mengapa Hasil Patung (Griffin) Muncul Dinding/Lempengan Raksasa?
* **Bukan Masalah Device/GPU:** GPU RTX 3050 (4GB) Anda berhasil menjalankan difusi 3D hingga tuntas (8 menit 12 detik tanpa crash).
* **Akar Masalah:**
  1. Pada foto studio (seperti patung Griffin di atas meja kayu dengan backdrop abu-abu), sistem pemotong latar belakang bawaan di Windows Python tidak memiliki modul AI `rembg` / `u2net`, sehingga jatuh ke algoritma *OpenCV GrabCut* klasik.
  2. GrabCut gagal membedakan backdrop studio dan bayangan meja dari patung (area yang dianggap foreground membengkak hingga **67.5%** dari total foto).
  3. GenSfM mengira dinding studio dan bayangan tersebut adalah bagian dari patung, sehingga merekonstruksinya menjadi **lempengan 3D vertikal raksasa** di samping patung.
  4. Pengujian model AI `rembg` (U2-Net) di WSL2 menunjukkan deteksi foreground hanya **18.38%** (tepat memotong patung dan tatakan kayu secara bersih, 100% dinding studio terhapus).

### B. Solusi 2 Jalur yang Diimplementasikan:
1. **Perbaikan Otomatis Internal (`masking.py`):**
   - Ditambahkan fungsi bridge otomatis `_run_ai_masking_wsl()`.
   - Jika node `GenSfM` dijalankan pada foto berlatar belakang studio tanpa node tambahan, sistem otomatis mendelegasikan proses segmentasi ke environment WSL2 AI (`rembg` / U2-Net) dalam ~2 detik per batch foto.
   - Algoritma GrabCut jadul tidak akan dipakai lagi selama WSL2 aktif.
2. **Node Baru di Canvas Meshroom: `AiRemoveBackground`:**
   - Dibuat node baru [`AiRemoveBackground.py`](file:///D:/Meshroom/mrGenSfM/meshroom/genSfm/AiRemoveBackground.py) pada kategori `Image Processing / AI`.
   - Node ini dapat dihubungkan di antara `CameraInit` dan `GenSfM` (`CameraInit.output -> AiRemoveBackground.input`, lalu `AiRemoveBackground.outputFolder -> GenSfM.inputMasks`).
   - Mendukung pemilihan model: `u2net` (*default/rekomendasi*), `birefnet-general`, `isnet-general-use`, `silueta`, dan `briaai-rmbg-2.0`.
   - Hasil potongan (mask & transparent RGBA) dapat langsung dipratinjau di **2D Image Viewer** Meshroom sebelum proses 3D dimulai.

### C. Analisis Lisensi Model:
* **U2-Net (`rembg` default):** Lisensi **Apache-2.0**. 100% bebas digunakan untuk tujuan komersial maupun pribadi/riset secara legal.
* **BRIA RMBG-2.0 (`briaai/RMBG-2.0`):** Lisensi **Non-Commercial Research**. Gratis untuk riset, edukasi, dan proyek personal; memerlukan lisensi berbayar jika digunakan untuk produk komersial enterprise.

### D. Catatan Kompatibilitas Node Descriptor Meshroom 2025.1.0:
* Meshroom 2025.1.0 memindai node secara modular menggunakan `pkgutil.iter_modules`. File `__init__.py` di dalam folder plugin package (`meshroom/genSfm/`) tidak boleh melakukan impor eager (`from .Node import Node`) agar tidak membayangi modul atau menyebabkan kegagalan discovery kelas.
* Parameter descriptor `desc.File`, `desc.ChoiceParam`, dan `desc.BoolParam` pada Meshroom 2025.1.0 tidak menerima argumen `uid=[0]` (UID dikelola secara otomatis oleh engine graph).

### E. Penanganan Line Ending Skrip WSL2 (CRLF vs LF):
* Skrip bash sementara (`_wsl_masking_runner.sh`) yang dibuat dari Windows untuk dieksekusi di WSL2 wajib ditulis dengan newline Unix (`\n` / LF) menggunakan `.write_bytes(sh_content.replace("\r\n", "\n").encode("utf-8"))`. Jika ditulis dengan CRLF Windows (`\r\n`), bash WSL akan memunculkan error `set: -: invalid option` dan `syntax error near unexpected token`.



