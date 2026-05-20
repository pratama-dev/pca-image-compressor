import streamlit as st
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os
import zipfile

# ──────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PCA Image Compressor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS KUSTOM
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0d6efd, #0dcaf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0d6efd;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<p class="main-title">🖼️ PCA Image Compressor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Kompresi Citra dengan Principal Component Analysis · Eigenvalue & Eigenvector · EDA</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FUNGSI UTAMA
# ──────────────────────────────────────────────

def pca_compress(channel: np.ndarray, k: int):
    """PCA compression pada satu channel citra."""
    mean        = np.mean(channel, axis=0)
    Xc          = channel - mean
    cov_matrix  = np.cov(Xc, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx  = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    W           = eigenvectors[:, :k]
    Z           = np.dot(Xc, W)
    reconstructed = np.dot(Z, W.T) + mean
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed, eigenvalues


def hitung_metrik(original: np.ndarray, rekonstruksi: np.ndarray):
    """Hitung MSE, PSNR, SSIM."""
    mse = np.mean((original - rekonstruksi) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)
    ssim_val = ssim(
        original.astype(np.uint8),
        rekonstruksi.astype(np.uint8),
        data_range=255,
    )
    return mse, psnr, ssim_val


def img_to_bytes(img_array: np.ndarray, fmt="JPEG") -> bytes:
    """Konversi numpy array ke bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


def ukuran_bytes(b: bytes) -> float:
    return len(b) / 1024  # KB


def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    uploaded_file = st.file_uploader(
        "Upload Citra", type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Format yang didukung: JPG, PNG, BMP, WebP"
    )

    st.markdown("---")
    st.markdown("**Nilai k (jumlah komponen utama)**")
    k_preset = st.multiselect(
        "Pilih nilai k yang akan diuji:",
        options=[1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200],
        default=[5, 10, 20, 50, 100],
        help="Semakin besar k, semakin baik kualitas rekonstruksi"
    )
    k_preset = sorted(k_preset)

    st.markdown("---")
    st.markdown("**Tampilan EDA Detail**")
    k_eda_count = st.slider("Jumlah k untuk EDA detail:", 1, min(5, max(len(k_preset), 1)), min(3, len(k_preset)))

    st.markdown("---")
    st.markdown("### 📖 Tentang Aplikasi")
    st.info(
        "Aplikasi ini mengimplementasikan kompresi citra menggunakan **PCA** "
        "berbasis **Eigenvalue & Eigenvector**. "
        "Dilengkapi **EDA** sebelum dan sesudah kompresi serta metrik evaluasi "
        "MSE, PSNR, SSIM, dan Rasio Kompresi."
    )

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 3rem; background:#f8f9fa; border-radius:12px; border: 2px dashed #0d6efd'>
            <h3>📤 Upload citra untuk memulai</h3>
            <p style='color:#6c757d'>Gunakan panel di sebelah kiri untuk mengupload gambar<br>
            Format: JPG · PNG · BMP · WebP</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

if not k_preset:
    st.warning("⚠️ Pilih minimal satu nilai k di sidebar.")
    st.stop()

# ──────────────────────────────────────────────
# BACA & PREP CITRA
# ──────────────────────────────────────────────
gambar      = Image.open(uploaded_file)
mode_citra  = "grayscale" if gambar.mode == "L" else "berwarna"
X_rgb       = np.array(gambar.convert("RGB"), dtype=np.float32)
X_gray      = np.array(gambar.convert("L"),   dtype=np.float32)

bytes_asli      = uploaded_file.getvalue()
ukuran_awal_kb  = ukuran_bytes(bytes_asli)

# Batasi k agar tidak melebihi dimensi citra
max_k       = X_gray.shape[1]
k_values    = [min(k, max_k) for k in k_preset]
k_values    = sorted(set(k_values))

# ──────────────────────────────────────────────
# TAB NAVIGASI
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Awal",
    "📈 Analisis Eigenvalue",
    "🗜️ Hasil Kompresi",
    "🔍 EDA Setelah Kompresi",
    "⬇️ Download",
])

# ══════════════════════════════════════════════
# TAB 1 – EDA AWAL
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">📊 EDA Awal – Sebelum Kompresi</p>', unsafe_allow_html=True)

    # Info umum
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode Citra",  mode_citra.upper())
    c2.metric("Ukuran",      f"{gambar.size[0]} × {gambar.size[1]} px")
    c3.metric("File Asli",   f"{ukuran_awal_kb:.1f} KB")
    c4.metric("Total Piksel",f"{gambar.size[0]*gambar.size[1]:,}")

    st.markdown("---")

    # Statistik grayscale
    col_stat, col_img = st.columns([1, 2])
    with col_stat:
        st.markdown("**Statistik Grayscale**")
        stats = {
            "Min Piksel"  : f"{X_gray.min():.0f}",
            "Max Piksel"  : f"{X_gray.max():.0f}",
            "Mean"        : f"{X_gray.mean():.2f}",
            "Std Deviation": f"{X_gray.std():.2f}",
        }
        for k_s, v in stats.items():
            st.markdown(f'<div class="metric-card"><b>{k_s}</b>: {v}</div>', unsafe_allow_html=True)

        if X_gray.mean() < 85:
            st.warning("🌑 Citra cenderung **GELAP**")
        elif X_gray.mean() > 170:
            st.success("☀️ Citra cenderung **TERANG**")
        else:
            st.info("🌤️ Kecerahan citra **NORMAL**")

        if X_gray.std() < 40:
            st.warning("📉 Kontras citra **RENDAH**")
        elif X_gray.std() > 80:
            st.success("📈 Kontras citra **TINGGI**")
        else:
            st.info("📊 Kontras citra **SEDANG**")

    with col_img:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(X_rgb.astype(np.uint8))
        axes[0].set_title("Citra Asli (RGB)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(X_gray, cmap='gray')
        axes[1].set_title("Citra Grayscale", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Histogram
    st.markdown("**Histogram Intensitas Piksel (Grayscale)**")
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.hist(X_gray.flatten(), bins=64, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(X_gray.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {X_gray.mean():.1f}')
    ax2.set_xlabel("Intensitas Piksel (0–255)")
    ax2.set_ylabel("Frekuensi")
    ax2.set_title("Histogram Piksel – Citra Asli")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 2 – ANALISIS EIGENVALUE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📈 Analisis Eigenvalue & Explained Variance</p>', unsafe_allow_html=True)

    with st.spinner("Menghitung eigenvalue…"):
        k_sample        = min(200, X_gray.shape[1])
        _, eigenvalues_all = pca_compress(X_gray, k_sample)

    ev_ratio    = eigenvalues_all / np.sum(eigenvalues_all)
    cumulative  = np.cumsum(ev_ratio)

    # Top 10
    st.markdown("**Top 10 Eigenvalue Terbesar**")
    cols = st.columns(5)
    for i in range(min(10, len(eigenvalues_all))):
        with cols[i % 5]:
            st.metric(f"PC{i+1}", f"{eigenvalues_all[i]:,.0f}")

    st.markdown("---")

    # Threshold
    st.markdown("**Jumlah Komponen (k) untuk Tiap Threshold**")
    thresh_cols = st.columns(4)
    for idx_t, thr in enumerate([0.80, 0.90, 0.95, 0.99]):
        k_needed = int(np.argmax(cumulative >= thr)) + 1
        thresh_cols[idx_t].metric(f"{int(thr*100)}% Variance", f"k = {k_needed}")

    st.markdown("---")
    col_cum, col_scree = st.columns(2)

    with col_cum:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(range(1, len(cumulative)+1), cumulative, color='steelblue', linewidth=2)
        ax3.axhline(0.90, color='orange', linestyle='--', label='90%')
        ax3.axhline(0.95, color='red',    linestyle='--', label='95%')
        ax3.axhline(0.99, color='green',  linestyle='--', label='99%')
        ax3.set_xlabel("Jumlah Komponen Utama (k)")
        ax3.set_ylabel("Cumulative Explained Variance")
        ax3.set_title("Cumulative Explained Variance")
        ax3.legend(); ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_scree:
        n_scree = min(30, len(eigenvalues_all))
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(range(1, n_scree+1), eigenvalues_all[:n_scree],
                 marker='o', color='darkorange', linewidth=2, markersize=5)
        ax4.set_xlabel("Komponen Utama (k)")
        ax4.set_ylabel("Eigenvalue")
        ax4.set_title(f"Scree Plot – {n_scree} Eigenvalue Terbesar")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

# ══════════════════════════════════════════════
# PROSES KOMPRESI (diperlukan tab 3, 4, 5)
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def jalankan_kompresi(gray_bytes: bytes, k_values_tuple: tuple):
    """Cache kompresi agar tidak dihitung ulang saat ganti tab."""
    X_g = np.array(Image.open(io.BytesIO(gray_bytes)).convert("L"), dtype=np.float32)
    k_sample = min(200, X_g.shape[1])
    _, eig_all = pca_compress(X_g, k_sample)
    ev_ratio_all = eig_all / np.sum(eig_all)

    hasil   = []
    metrik  = []
    for k in k_values_tuple:
        k_eff   = min(k, X_g.shape[1])
        recon, _= pca_compress(X_g, k_eff)
        recon8  = recon.astype(np.uint8)
        mse, psnr_val, ssim_val = hitung_metrik(X_g, recon)

        ev_cum  = float(np.sum(ev_ratio_all[:k_eff])) * 100

        b_recon = img_to_bytes(recon8, "JPEG")
        uk_kb   = ukuran_bytes(b_recon)

        hasil.append(recon8)
        metrik.append({
            "k"          : k_eff,
            "ev"         : ev_cum,
            "mse"        : mse,
            "psnr"       : psnr_val,
            "ssim"       : ssim_val,
            "ukuran_kb"  : uk_kb,
            "bytes"      : b_recon,
        })
    return hasil, metrik, X_g, eig_all


with st.spinner("⏳ Memproses kompresi PCA untuk semua nilai k…"):
    bytes_gray      = uploaded_file.getvalue()
    hasil_gray, metrik_list, X_gray_cached, eig_all_cached = jalankan_kompresi(
        bytes_gray, tuple(k_values)
    )
    ukuran_awal_kb_ref = ukuran_awal_kb


# ══════════════════════════════════════════════
# TAB 3 – HASIL KOMPRESI
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">🗜️ Hasil Kompresi PCA – Semua Nilai k</p>', unsafe_allow_html=True)

    # Tabel metrik
    st.markdown("**Tabel Evaluasi Lengkap**")
    import pandas as pd
    rows = []
    for m in metrik_list:
        penghematan = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
        cr = ukuran_awal_kb_ref / m["ukuran_kb"] if m["ukuran_kb"] > 0 else 0
        rows.append({
            "k"                  : m["k"],
            "Expl. Var (%)"      : f"{m['ev']:.2f}",
            "MSE"                : f"{m['mse']:.2f}",
            "PSNR (dB)"          : f"{m['psnr']:.2f}",
            "SSIM"               : f"{m['ssim']:.4f}",
            "Ukuran (KB)"        : f"{m['ukuran_kb']:.2f}",
            "Penghematan (%)"    : f"{penghematan:.1f}",
            "CR (x)"             : f"{cr:.2f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Visualisasi Berdampingan**")

    # Grid citra hasil
    n_show = len(k_values)
    cols_img = st.columns(min(n_show + 1, 5))

    def show_img_col(col, img_arr, title, subtitle=""):
        with col:
            st.image(img_arr.astype(np.uint8), caption=title, use_container_width=True)
            if subtitle:
                st.caption(subtitle)

    # Tampilkan citra asli di kolom pertama (atau lebih kolom jika banyak k)
    chunk_size = 4
    groups = [k_values[i:i+chunk_size] for i in range(0, len(k_values), chunk_size)]

    for grp in groups:
        cols_g = st.columns(len(grp) + 1)
        with cols_g[0]:
            st.image(X_gray_cached.astype(np.uint8), caption="Citra Asli", use_container_width=True,
                     clamp=True)
            st.caption(f"Ukuran: {ukuran_awal_kb_ref:.1f} KB")
        for j, kv in enumerate(grp):
            idx = k_values.index(kv)
            m   = metrik_list[idx]
            with cols_g[j+1]:
                st.image(hasil_gray[idx], caption=f"k = {kv}", use_container_width=True)
                st.caption(f"PSNR: {m['psnr']:.1f} dB | SSIM: {m['ssim']:.3f}")

    st.markdown("---")
    st.markdown("**Grafik Metrik Evaluasi**")

    ks      = [m["k"]    for m in metrik_list]
    mses    = [m["mse"]  for m in metrik_list]
    psnrs   = [m["psnr"] for m in metrik_list]
    ssims   = [m["ssim"] for m in metrik_list]
    ukurans = [m["ukuran_kb"] for m in metrik_list]

    fig5, axes5 = plt.subplots(2, 2, figsize=(13, 8))
    axes5[0,0].plot(ks, mses,   marker='o', color='tomato',     linewidth=2)
    axes5[0,0].set_title("MSE vs k"); axes5[0,0].set_xlabel("k"); axes5[0,0].set_ylabel("MSE"); axes5[0,0].grid(True, alpha=0.3)

    axes5[0,1].plot(ks, psnrs,  marker='s', color='steelblue',  linewidth=2)
    axes5[0,1].axhline(30, color='orange', linestyle='--', label='30 dB (Baik)')
    axes5[0,1].axhline(40, color='green',  linestyle='--', label='40 dB (Sangat Baik)')
    axes5[0,1].set_title("PSNR vs k"); axes5[0,1].set_xlabel("k"); axes5[0,1].set_ylabel("PSNR (dB)")
    axes5[0,1].legend(fontsize=8); axes5[0,1].grid(True, alpha=0.3)

    axes5[1,0].plot(ks, ssims,  marker='^', color='seagreen',   linewidth=2)
    axes5[1,0].axhline(0.90, color='red', linestyle='--', label='SSIM=0.90')
    axes5[1,0].set_title("SSIM vs k"); axes5[1,0].set_xlabel("k"); axes5[1,0].set_ylabel("SSIM")
    axes5[1,0].set_ylim(0, 1.05); axes5[1,0].legend(fontsize=8); axes5[1,0].grid(True, alpha=0.3)

    axes5[1,1].plot(ks, ukurans, marker='D', color='darkorange', linewidth=2)
    axes5[1,1].axhline(ukuran_awal_kb_ref, color='navy', linestyle='--',
                       label=f'Asli = {ukuran_awal_kb_ref:.1f} KB')
    axes5[1,1].set_title("Ukuran File vs k"); axes5[1,1].set_xlabel("k"); axes5[1,1].set_ylabel("Ukuran (KB)")
    axes5[1,1].legend(fontsize=8); axes5[1,1].grid(True, alpha=0.3)

    plt.suptitle("Evaluasi Kompresi PCA – Semua Nilai k", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)


# ══════════════════════════════════════════════
# TAB 4 – EDA SETELAH KOMPRESI
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">🔍 EDA Setelah Kompresi – Detail per k</p>', unsafe_allow_html=True)

    n_eda   = min(k_eda_count, len(k_values))
    step    = max(1, len(k_values) // n_eda)
    idx_eda = list(dict.fromkeys(
        [0] + list(range(0, len(k_values), step)) + [len(k_values)-1]
    ))[:n_eda]

    for idx in idx_eda:
        k_cur   = metrik_list[idx]["k"]
        m       = metrik_list[idx]
        img_rec = hasil_gray[idx]

        st.markdown(f"#### k = {k_cur}  |  PSNR = {m['psnr']:.2f} dB  |  SSIM = {m['ssim']:.4f}")

        fig6, axes6 = plt.subplots(1, 4, figsize=(18, 4))

        axes6[0].imshow(X_gray_cached.astype(np.uint8), cmap='gray')
        axes6[0].set_title("Citra Asli"); axes6[0].axis('off')

        axes6[1].imshow(img_rec, cmap='gray')
        axes6[1].set_title(f"Rekonstruksi k={k_cur}\nPSNR={m['psnr']:.2f} | SSIM={m['ssim']:.4f}")
        axes6[1].axis('off')

        err = np.abs(X_gray_cached - img_rec.astype(np.float32))
        im6 = axes6[2].imshow(err, cmap='hot')
        axes6[2].set_title(f"Error Image\nMSE={m['mse']:.2f}"); axes6[2].axis('off')
        plt.colorbar(im6, ax=axes6[2], fraction=0.046)

        axes6[3].hist(X_gray_cached.flatten(), bins=64, alpha=0.6, color='steelblue', label='Asli')
        axes6[3].hist(img_rec.flatten().astype(float), bins=64, alpha=0.6, color='tomato', label=f'k={k_cur}')
        axes6[3].set_xlabel("Intensitas"); axes6[3].set_ylabel("Frekuensi")
        axes6[3].set_title("Perbandingan Histogram"); axes6[3].legend(fontsize=8)

        plt.suptitle(f"EDA Setelah Kompresi – k = {k_cur}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)
        st.markdown("---")


# ══════════════════════════════════════════════
# TAB 5 – DOWNLOAD
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">⬇️ Download Hasil Kompresi</p>', unsafe_allow_html=True)

    st.markdown("**Download per nilai k**")
    dl_cols = st.columns(min(len(k_values), 4))
    for i, (kv, m) in enumerate(zip(k_values, metrik_list)):
        with dl_cols[i % 4]:
            penghematan = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
            st.download_button(
                label       = f"📥 k={kv}  ({m['ukuran_kb']:.1f} KB, -{penghematan:.0f}%)",
                data        = m["bytes"],
                file_name   = f"PCA_compressed_k{kv}.jpg",
                mime        = "image/jpeg",
                key         = f"dl_{kv}",
            )

    st.markdown("---")
    st.markdown("**Download Semua Sekaligus (ZIP)**")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for kv, m in zip(k_values, metrik_list):
            zf.writestr(f"PCA_compressed_k{kv}.jpg", m["bytes"])

        # Tambahkan laporan teks
        report_lines = [
            "LAPORAN KOMPRESI PCA",
            "=" * 60,
            f"Ukuran Citra   : {gambar.size[0]} x {gambar.size[1]} px",
            f"File Asli      : {ukuran_awal_kb_ref:.2f} KB",
            "",
            f"{'k':>5} | {'Expl.Var':>9} | {'MSE':>8} | {'PSNR':>9} | {'SSIM':>6} | {'Ukuran':>8} | {'Hemat':>6}",
            "-" * 65,
        ]
        for m in metrik_list:
            ph = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
            report_lines.append(
                f"{m['k']:>5} | {m['ev']:>8.2f}% | {m['mse']:>8.2f} | {m['psnr']:>8.2f} | {m['ssim']:>6.4f} | {m['ukuran_kb']:>6.2f}KB | {ph:>5.1f}%"
            )
        zf.writestr("laporan_kompresi.txt", "\n".join(report_lines))

    zip_buf.seek(0)
    st.download_button(
        label       = "📦 Download Semua Hasil (.zip)",
        data        = zip_buf.getvalue(),
        file_name   = "PCA_compressed_all.zip",
        mime        = "application/zip",
        key         = "dl_zip",
    )

    st.markdown("---")
    st.markdown("**Ringkasan Rekomendasi k Optimal**")
    best_idx = max(range(len(metrik_list)), key=lambda i: metrik_list[i]["ssim"] * 0.5 + (metrik_list[i]["psnr"] / 50) * 0.3 - (metrik_list[i]["ukuran_kb"] / ukuran_awal_kb_ref) * 0.2)
    bm = metrik_list[best_idx]
    st.success(
        f"✅ **k = {bm['k']}** direkomendasikan sebagai titik optimal  \n"
        f"Explained Variance: **{bm['ev']:.1f}%** · PSNR: **{bm['psnr']:.1f} dB** · "
        f"SSIM: **{bm['ssim']:.4f}** · Ukuran: **{bm['ukuran_kb']:.1f} KB**"
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:0.8rem;'>"
    "PCA Image Compressor · Eigenvalue & Eigenvector · EDA"
    "</p>",
    unsafe_allow_html=True,
)
