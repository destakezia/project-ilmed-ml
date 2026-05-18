# 📋 PROJECT CONTEXT — ILMU DATA MEDIS (ILMED)
> Dokumen ini dibuat sebagai starter prompt untuk sesi AI selanjutnya.
> Lampirkan dokumen ini + file lampiran yang disebutkan di bagian bawah setiap kali memulai sesi baru.

---

## 🎓 Informasi Akademik

| Atribut | Detail |
|---|---|
| Mata Kuliah | Ilmu Data Medis (ILMED) - Kelas C |
| Institusi | Institut Teknologi Sepuluh Nopember (ITS) |
| Jenis Tugas | Project Akhir Semester |
| Jumlah Anggota | Maksimal 4 orang per kelompok |
| Progress | Monitoring mingguan |

---

## 🎯 Judul Project

**"Klasifikasi Multiclass Penyakit Paru pada Citra Chest X-Ray menggunakan Arsitektur CNN dengan Optimasi Data Balancing berbasis Generative Adversarial Networks (GAN) dan Analisis Interpretability Grad-CAM"**

---

## 📂 Dataset

- **Nama**: Chest X-Ray 6 Classes Dataset
- **Sumber**: Kaggle
- **Link**: https://www.kaggle.com/datasets/mohamedalisak/chest-x-ray-6-classes-dataset
- **Jumlah Kelas**: 6 kelas (multiclass)
- **Format**: Gambar (JPG/PNG)
- **Split yang Direncanakan**: 80% Train / 20% Validation (atau 70:30)

> ⚠️ Dataset ini sudah dikonfirmasi berbeda dari kelompok lain di kelas.

---

## 📄 Referensi Paper Utama

- **Judul**: LungNet22: A Fine-Tuned Model for Multiclass Classification and Prediction of Lung Disease Using X-ray Images
- **Penulis**: F. M. Javed Mehedi Shamrat, Sami Azam, Asif Karim, Rakibul Islam, Zarrin Tasnim, Pronab Ghosh, Friso De Boer
- **Jurnal**: Journal of Personalized Medicine (MDPI), 2022, Vol. 12, No. 680
- **DOI**: https://doi.org/10.3390/jpm12050680
- **GitHub Kode**: https://github.com/Shamrat777/Lung-Disease-Classiffication
- **Supplementary Materials** (video tutorial + runnable code + sample dataset): https://www.mdpi.com/article/10.3390/jpm12050680/s1

> ⚠️ Kode GitHub dibuat 2022 — kemungkinan perlu update versi library (TensorFlow/Keras).

---

## 🔬 Metode yang Digunakan

### Arsitektur Utama: LungNet22 (Fine-tuned VGG16)

Paper asli menggunakan VGG16 sebagai backbone, lalu ditambah 2 block baru:
- **Block 6**: 3x Conv2D (512 channel, kernel 3×3, ReLU) + GlobalAveragePooling2D
- **Block 7**: Flatten + Dense(jumlah_kelas) + Softmax

**Modifikasi untuk project ini**:
- Output layer: `Dense(10)` → `Dense(6)` (karena 6 kelas)
- Dataset path disesuaikan dengan dataset Kaggle baru
- Tambahan GAN untuk data balancing (tidak ada di paper asli)

### Hyperparameter Terbaik (dari Ablation Study di Paper)

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.000001 |
| Loss Function | Categorical Crossentropy |
| Pooling Layer | GlobalAveragePooling2D |
| Input Size | 224 × 224 × 3 pixel |
| Batch Size | (sesuaikan dengan RAM) |
| Epochs | 100–300 |
| Data Split | 80:20 atau 70:30 |

---

## 🔄 Pipeline Lengkap Project

```
TAHAP 1 — EKSPLORASI DATA
├── Download dataset dari Kaggle
├── Hitung distribusi per kelas (total, train, val)
└── Visualisasi sample gambar per kelas

TAHAP 2 — PREPROCESSING
├── Image Denoising (Denoising Autoencoder / DAE)
├── Annotation Removal (EnsNet) — opsional jika ada teks di gambar
├── Image Enhancement (CLAHE)
│     └── tileGridSize=(10,10), clipLimit=3
└── Filter Application (Green Fire Blue filter)

TAHAP 3 — DATA BALANCING & AUGMENTASI
├── Cek distribusi kelas (apakah imbalanced?)
├── Undersampling: kelas mayoritas → max 8000 data
├── Augmentasi standar: Rotate, Flip, Crop, Translate, Zoom
└── GAN untuk generate data sintetis kelas minoritas
      └── Rekomendasi: Conditional GAN (cGAN) atau DCGAN

TAHAP 4 — DUA SKENARIO (WAJIB dari instruksi dosen)
├── Skenario A: Imbalanced (data asli tanpa GAN)
└── Skenario B: Balanced (dengan GAN + augmentasi)
      → Bandingkan hasil keduanya!

TAHAP 5 — TRAINING MODEL
├── Transfer Learning: VGG16 pretrained (ImageNet)
├── Fine-tuning: tambah Block 6 & Block 7
├── Training dengan Adam optimizer, LR=0.000001
└── Simpan model terbaik (checkpoint)

TAHAP 6 — EVALUASI (semua wajib ditampilkan)
├── Grafik Train/Val Accuracy & Loss per epoch
├── Confusion Matrix (True vs Predicted label)
├── Metrics per kelas:
│     Accuracy, Precision, Recall (Sensitivity),
│     Specificity, F1-Score
├── ROC Curve + AUC per kelas
└── Grad-CAM + Score-CAM visualization
```

---

## ✅ Checklist Output Wajib (dari Instruksi Dosen)

- [ ] Tabel distribusi data (per kelas: total, train, val)
- [ ] Grafik distribusi data (bar chart)
- [ ] Visualisasi sample citra per kelas
- [ ] Ilustrasi flow/arsitektur metode CNN
- [ ] Grafik akurasi & loss vs epoch (train + val)
- [ ] Confusion Matrix
- [ ] Grafik batang metrics (Accuracy, Precision, Recall, Specificity, F1)
- [ ] ROC Curve + AUC
- [ ] Grad-CAM visualization
- [ ] Score-CAM visualization (opsional, ada di contoh PPT dosen)
- [ ] Perbandingan Balanced vs Imbalanced

---

## 💻 Environment & Tools

### Opsi 1: Google Colab (Rekomendasi untuk mulai)
- Gratis, GPU T4
- Tidak perlu install
- Limit sesi ~12 jam (simpan model rutin ke Google Drive)
- Colab Pro (~Rp60rb/bulan) untuk GPU lebih kuat (disarankan untuk GAN)

### Opsi 2: VS Code + Local
- Install: Python 3.9+, CUDA (jika ada GPU NVIDIA)
- Library: TensorFlow/Keras, PyTorch (pilih salah satu)
- Perlu GPU NVIDIA minimal 6GB VRAM untuk training nyaman

### Library yang Dibutuhkan
```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
Pillow
grad-cam (pytorch-grad-cam atau tf-keras-vis)
```

---

## 📦 Modifikasi Kode dari GitHub Paper

File yang **wajib dimodifikasi** dari repo Shamrat777:

| File/Bagian | Perubahan |
|---|---|
| Jumlah kelas output | `Dense(10)` → `Dense(6)` |
| Path dataset | Ganti ke path dataset Kaggle baru |
| Class names/labels | Sesuaikan dengan 6 kelas baru |
| Data split ratio | 60:20:20 → 80:20 atau 70:30 |
| Epochs | 300 → 50–100 (sesuaikan komputasi) |

File yang **perlu ditambahkan** (tidak ada di paper):
- Kode GAN (DCGAN atau cGAN) untuk generate data sintetis
- Score-CAM (paper hanya ada Grad-CAM)
- Skenario perbandingan Balanced vs Imbalanced

---

## 📁 Lampiran yang Harus Disertakan di Setiap Sesi AI

Selalu lampirkan file-file berikut saat memulai sesi baru:

### Lampiran Wajib
1. **File ini** (`PROJECT_CONTEXT_ILMED.md`) → sebagai context utama
2. **Paper LungNet22** (`jpm-12-00680.pdf`) → referensi metode utama
3. **PPT Contoh Dosen** (`06-Data_Sains_-_Project.pdf`) → format output yang diharapkan

### Lampiran Opsional (sesuai kebutuhan sesi)
4. Kode yang sedang dikerjakan (`.py` atau `.ipynb`)
5. Screenshot error jika ada bug
6. File dataset sample (jika perlu diskusi struktur data)

### Link Penting (bookmark ini)
- Dataset Kaggle: https://www.kaggle.com/datasets/mohamedalisak/chest-x-ray-6-classes-dataset
- GitHub kode paper: https://github.com/Shamrat777/Lung-Disease-Classiffication
- Supplementary materials: https://www.mdpi.com/article/10.3390/jpm12050680/s1
- DOI Paper: https://doi.org/10.3390/jpm12050680

---

## 🗓️ Timeline Progress (Estimasi)

| Minggu | Target |
|---|---|
| Minggu 1 (sekarang) | Eksplorasi dataset, distribusi data, visualisasi sample, setup environment |
| Minggu 2 | Preprocessing (CLAHE, resize, normalisasi), augmentasi standar |
| Minggu 3 | Implementasi GAN untuk data balancing |
| Minggu 4 | Training model CNN (Skenario Imbalanced dulu) |
| Minggu 5 | Training model CNN (Skenario Balanced + GAN) |
| Minggu 6 | Evaluasi: confusion matrix, metrics, ROC curve |
| Minggu 7 | Grad-CAM + Score-CAM, perbandingan skenario |
| Minggu 8 | Pembuatan laporan + slide presentasi final |

---

## 💡 Catatan Tambahan untuk AI

- Bahasa komunikasi: **Bahasa Indonesia** (campur English untuk istilah teknis)
- Level pemahaman user: **mahasiswa**, familiar dengan Python dasar
- Prioritas: **step by step**, jangan skip penjelasan
- Jika memberikan kode, selalu sertakan **komentar penjelasan**
- Dataset ini adalah gambar **Chest X-Ray** (bukan sel darah/parasit)
- Project ini untuk mata kuliah **Ilmu Data Medis di ITS**, bukan penelitian mandiri
- Paper referensi menggunakan **10 kelas**, project ini hanya **6 kelas**
- GAN **TIDAK ADA** di paper referensi — harus diimplementasi sendiri
- Contoh output yang diharapkan dosen ada di **PPT file dosen** (Vision Transformer dengan Score-CAM sebagai referensi visual format, bukan metodenya)

---

*Dokumen dibuat: Mei 2026 | Versi 1.0*
