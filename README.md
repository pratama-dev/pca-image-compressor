# ✦ PCA Image Compressor

> Kompresi citra berbasis **Principal Component Analysis** menggunakan Eigenvalue & Eigenvector yang dilengkapi analisis EDA mendalam sebelum dan sesudah kompresi, dengan nilai k yang sepenuhnya dapat dikustomisasi oleh pengguna.

![Python](https://img.shields.io/badge/Python-3.9+-a78bfa?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-7c3aed?style=flat-square&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-6d28d9?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## 📸 Tentang Proyek

**PCA Image Compressor** adalah aplikasi web interaktif yang mengimplementasikan teknik kompresi citra menggunakan *Principal Component Analysis (PCA)* berbasis **Eigenvalue & Eigenvector**. Aplikasi ini dirancang tidak hanya sebagai alat kompresi, tetapi juga sebagai platform analitik visual yang lengkap. Mulai dari EDA awal, analisis dekomposisi eigen, evaluasi metrik kualitas, hingga perbandingan hasil kompresi untuk berbagai nilai k.

Pengguna bebas menentukan sendiri nilai-nilai k yang ingin diuji cukup dengan mengetiknya di sidebar, tanpa batasan pilihan yang sudah ditetapkan.

Proyek ini dibuat sebagai implementasi praktis dari materi **Kompresi Citra dengan PCA** dalam konteks mata kuliah Aljabar Linear / Data Science.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| ✏️ **Input K Bebas** | Ketik nilai k sendiri dipisah koma — bebas tentukan berapa saja yang ingin diuji |
| 📊 **EDA Awal** | Statistik piksel, histogram intensitas, deteksi kecerahan & kontras otomatis |
| 📈 **Analisis Eigenvalue** | Scree Plot, Cumulative Explained Variance, threshold otomatis (80/90/95/99%) |
| 🗜️ **Kompresi Multi-k** | Uji semua nilai k yang diinput sekaligus dalam satu proses |
| 🔍 **EDA Pasca Kompresi** | Error Map (plasma colormap), perbandingan histogram, metrik per k |
| 📦 **Export & Download** | Download per k (JPG) atau semua sekaligus + laporan teks dalam satu ZIP |
| ✦ **Rekomendasi Optimal** | Deteksi otomatis nilai k terbaik berdasarkan skor gabungan SSIM, PSNR, dan ukuran file |

---

## ✏️ Cara Input Nilai K

Di sidebar kiri, terdapat kolom teks **"Nilai K — Input Manual"**. Cukup ketik angka-angka yang diinginkan dipisah dengan koma:

```
5, 10, 20, 50, 100
```

**Aturan input:**
- Nilai harus berupa **bilangan bulat positif** (≥ 1)
- Nilai maksimum adalah **2000** per entri
- Jika nilai k melebihi dimensi citra, otomatis dibatasi sesuai ukuran citra
- Duplikat otomatis dihapus, urutan otomatis diurutkan

**Contoh input valid:**
| Input | Hasil |
|-------|-------|
| `5, 10, 20` | k = 5, 10, 20 |
| `1, 50, 100, 200` | k = 1, 50, 100, 200 |
| `10, 10, 30` | k = 10, 30 *(duplikat dihapus)* |
| `100` | k = 100 *(satu nilai pun boleh)* |

Sidebar menampilkan **konfirmasi hijau** jika input valid, atau **pesan error merah** jika ada kesalahan format.

---

## 📐 Metrik Evaluasi

| Metrik | Keterangan | Nilai Baik |
|--------|-----------|------------|
| **MSE** | Mean Squared Error — rata-rata kuadrat selisih piksel | Sekecil mungkin |
| **PSNR** | Peak Signal-to-Noise Ratio — kualitas berbasis error | > 30 dB (baik), > 40 dB (sangat baik) |
| **SSIM** | Structural Similarity Index — kemiripan struktur visual | Mendekati 1.0 |
| **CR** | Compression Ratio — efisiensi pengurangan ukuran | Semakin besar semakin efisien |

---

## 🏗️ Alur Kompresi PCA

```
Citra Asli
    │
    ▼
EDA Awal (histogram, statistik)
    │
    ▼
Konversi → Matriks [m × n]
    │
    ▼
Centering:  Xc = X - μ
    │
    ▼
Matriks Kovarians:  C = (1/n-1) · Xc^T · Xc
    │
    ▼
Eigen Decomposition:  C·v = λ·v
    │
    ▼
Sort eigenvalue (terbesar → terkecil)
    │
    ▼
Pilih k eigenvector terbaik  →  W_k
    │
    ▼
Proyeksi:  Z = Xc · W_k
    │
    ▼
Rekonstruksi:  X̂ = Z · W_k^T + μ
    │
    ▼
Evaluasi (MSE · PSNR · SSIM · CR)
    │
    ▼
EDA Pasca Kompresi
```

---

## 🚀 Instalasi & Menjalankan Lokal

### Prasyarat
- Python 3.9 atau lebih baru
- pip

### Langkah-langkah

```bash
# 1. Clone repositori
git clone https://github.com/pratama-dev/pca-compressor.git
cd pca-compressor

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Community Cloud

**Gratis dan Tidak perlu server sendiri.**

### Langkah Deploy

1. **Buat repo GitHub baru** bernama `pca-compressor`

2. **Upload 2 file** ke root repo:
   - `app.py`
   - `requirements.txt`

3. **Buka** [share.streamlit.io](https://share.streamlit.io) → login dengan GitHub

4. **Klik "New app"** dan isi:
   ```
   Repository  : username/pca-compressor
   Branch      : main
   Main file   : app.py
   ```

5. **Klik Deploy** → tunggu ~2 menit

6. **Selesai!** URL publik aktif:
   ```
   https://username-pca-compressor.streamlit.app
   ```

---

## 📁 Struktur Proyek

```
pca-compressor/
├── app.py              # Aplikasi Streamlit utama
├── requirements.txt    # Daftar dependencies Python
└── README.md           # Dokumentasi ini
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-image>=0.21.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

## 🎨 Design System

Aplikasi menggunakan tema **Obsidian Dark Mode** dengan aksen **Electric Violet**:

| Token | Nilai | Keterangan |
|-------|-------|-----------|
| `BG_BASE` | `#09090b` | Latar belakang utama |
| `ACCENT` | `#7c3aed` | Warna aksen utama |
| `ACCENT_LIGHT` | `#a78bfa` | Aksen terang untuk teks |
| `SUCCESS` | `#22c55e` | Indikator positif / konfirmasi |
| `DANGER` | `#ef4444` | Indikator error / peringatan |
| Font | Inter | Google Fonts |

Efek visual: **Glassmorphism** · **Hover animation** · **Gradient fill** pada semua grafik.

---

## 💡 Panduan Penggunaan

1. **Upload citra** (JPG/PNG/BMP/WebP) via sidebar kiri
2. **Ketik nilai k** yang ingin diuji, pisah dengan koma — misal: `5, 10, 20, 50, 100`
3. Pastikan muncul **konfirmasi hijau** ✓ di bawah kolom input
4. Buka tab **EDA Awal** untuk memahami karakteristik citra
5. Buka tab **Eigenvalue** untuk melihat Scree Plot dan threshold optimal
6. Buka tab **Kompresi** untuk membandingkan hasil berbagai nilai k
7. Buka tab **EDA Pasca** untuk analisis Error Map & histogram
8. Buka tab **Export** untuk download hasil + laporan otomatis

---

## 📊 Contoh Hasil

| k | Expl. Var | MSE | PSNR | SSIM | Hemat |
|---|-----------|-----|------|------|-------|
| 5 | ~65% | Tinggi | ~22 dB | ~0.70 | ~85% |
| 20 | ~88% | Sedang | ~31 dB | ~0.88 | ~60% |
| 50 | ~96% | Rendah | ~37 dB | ~0.94 | ~40% |
| 100 | ~99% | Sangat rendah | ~42 dB | ~0.98 | ~20% |

*Nilai aktual bergantung pada karakteristik citra yang diupload.*

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan untuk keperluan akademik maupun komersial.

---

<div align="center">
  <strong>✦ PCA Image Compressor</strong><br>
  Dibuat dengan ♥ menggunakan Python · NumPy · Streamlit
</div>
