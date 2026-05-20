import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os
import zipfile
# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="PCA Image Compressor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 2.75rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563eb, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Upload Area Styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        background-color: #f8fafc;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6;
        background-color: #eff6ff;
    }
    
    /* Tabs Styling */
    [data-testid="stTabs"] button {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)
# HEADER

st.markdown('<p class="main-title">PCA Image Compressor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Kompresi Citra Berbasis Principal Component Analysis & Exploratory Data Analysis</p>', unsafe_allow_html=True)


def pca_compress(channel: np.ndarray, k: int):
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
    mse = np.mean((original - rekonstruksi) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)
    ssim_val = ssim(
        original.astype(np.uint8),
        rekonstruksi.astype(np.uint8),
        data_range=255,
    )
    return mse, psnr, ssim_val

def img_to_bytes(img_array: np.ndarray, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()

def ukuran_bytes(b: bytes) -> float:
    return len(b) / 1024  # KB

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8364/8364801.png", width=60)
    st.markdown("### ⚙️ Konfigurasi")
    
    uploaded_file = st.file_uploader(
        "Upload Citra Analisis", type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Mendukung format: JPG, PNG, BMP, WebP"
    )

    st.markdown("---")
    
    with st.expander("🛠️ Parameter PCA", expanded=True):
        st.markdown("**Nilai *k* (Principal Components)**")
        k_preset = st.multiselect(
            "Pilih nilai *k* untuk evaluasi:",
            options=[1, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200],
            default=[5, 10, 20, 50, 100],
            help="Menentukan jumlah vektor eigen yang dipertahankan."
        )
        k_preset = sorted(k_preset)
        
        k_eda_count = st.slider("Batas tampilan visual EDA:", 1, min(5, max(len(k_preset), 1)), min(3, len(k_preset)))

    st.markdown("---")
    st.info(
        "**Tentang Sistem**\n\n"
        "Implementasi kompresi citra reduksi dimensi menggunakan dekomposisi matriks kovarians, "
        "dilengkapi metrik evaluasi MSE, PSNR, dan SSIM."
    )

# ──────────────────────────────────────────────
# MAIN CONTENT / EMPTY STATE
# ──────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center; padding: 5rem 2rem; background: linear-gradient(145deg, #ffffff, #f8fafc); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-top: 2rem;'>
        <h2 style='color: #0f172a; margin-bottom: 1rem; font-weight: 600;'>Mulai Analisis Kompresi PCA</h2>
        <p style='color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>Silakan unggah citra melalui panel di sebelah kiri untuk menginisiasi proses ekstraksi fitur dan reduksi dimensi.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not k_preset:
    st.warning("⚠️ Parameter tidak valid: Pilih minimal satu nilai *k* pada panel konfigurasi.")
    st.stop()

# PEMROSESAN CITRA
gambar      = Image.open(uploaded_file)
mode_citra  = "Grayscale" if gambar.mode == "L" else "RGB / Berwarna"
X_rgb       = np.array(gambar.convert("RGB"), dtype=np.float32)
X_gray      = np.array(gambar.convert("L"),   dtype=np.float32)

bytes_asli      = uploaded_file.getvalue()
ukuran_awal_kb  = ukuran_bytes(bytes_asli)

max_k       = X_gray.shape[1]
k_values    = sorted(set([min(k, max_k) for k in k_preset]))
# STRUKTUR TAB

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Profil Citra",
    "📈 Analisis Eigenvalue",
    "🗜️ Evaluasi Kompresi",
    "🔍 Analisis Spasial",
    "📥 Ekspor Hasil",
])

# TAB 1 – EDA AWAL

with tab1:
    st.markdown('<p class="section-header">Profil Citra Original</p>', unsafe_allow_html=True)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Mode Ruang Warna</div><div class="metric-value">{mode_citra}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Resolusi Spasial</div><div class="metric-value">{gambar.size[0]} × {gambar.size[1]} px</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Ukuran Memori</div><div class="metric-value">{ukuran_awal_kb:.2f} KB</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Piksel</div><div class="metric-value">{gambar.size[0]*gambar.size[1]:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_stat, col_img = st.columns([1, 2.5])
    with col_stat:
        st.markdown("**Statistik Distribusi Intensitas**")
        stats = {
            "Min Intensitas": f"{X_gray.min():.0f}",
            "Max Intensitas": f"{X_gray.max():.0f}",
            "Mean (μ)": f"{X_gray.mean():.2f}",
            "Std Dev (σ)": f"{X_gray.std():.2f}",
        }
        for k_s, v in stats.items():
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid #f1f5f9;">
                <span style="color: #64748b; font-weight: 500;">{k_s}</span>
                <span style="color: #0f172a; font-weight: 600;">{v}</span>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if X_gray.mean() < 85:
            st.info("💡 **Karakteristik**: Dominasi area low-key (Cenderung Gelap)")
        elif X_gray.mean() > 170:
            st.info("💡 **Karakteristik**: Dominasi area high-key (Cenderung Terang)")
        
        if X_gray.std() < 40:
            st.info("💡 **Kontras**: Rentang dinamis rendah (Low Contrast)")
        elif X_gray.std() > 80:
            st.info("💡 **Kontras**: Rentang dinamis tinggi (High Contrast)")

    with col_img:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].imshow(X_rgb.astype(np.uint8))
        axes[0].set_title("Citra Original (RGB)", pad=15, fontweight='bold', color='#1e293b')
        axes[0].axis('off')
        
        axes[1].imshow(X_gray, cmap='gray')
        axes[1].set_title("Representasi Grayscale", pad=15, fontweight='bold', color='#1e293b')
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Histogram Intensitas Piksel (Grayscale)**")
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    ax2.hist(X_gray.flatten(), bins=256, color='#3b82f6', alpha=0.8)
    ax2.axvline(X_gray.mean(), color='#ef4444', linestyle='dashed', linewidth=2, label=f'Mean (μ) = {X_gray.mean():.1f}')
    ax2.set_xlabel("Nilai Intensitas (0-255)", color='#64748b')
    ax2.set_ylabel("Frekuensi", color='#64748b')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cbd5e1')
    ax2.spines['bottom'].set_color('#cbd5e1')
    ax2.legend(frameon=False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# TAB 2 – ANALISIS EIGENVALUE
with tab2:
    st.markdown('<p class="section-header">Analisis Spektrum Matriks Kovarians</p>', unsafe_allow_html=True)

    with st.spinner("Mengekstraksi Vektor Eigen..."):
        k_sample = min(200, X_gray.shape[1])
        _, eigenvalues_all = pca_compress(X_gray, k_sample)

    ev_ratio    = eigenvalues_all / np.sum(eigenvalues_all)
    cumulative  = np.cumsum(ev_ratio)

    st.markdown("**Top 5 Principal Components Berdasarkan Nilai Eigen**")
    cols = st.columns(5)
    for i in range(min(5, len(eigenvalues_all))):
        with cols[i]:
            st.metric(label=f"PC {i+1}", value=f"{eigenvalues_all[i]:,.0f}", delta=f"{(ev_ratio[i]*100):.2f}% Var", delta_color="normal")

    st.markdown("<hr style='margin: 2rem 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)
    
    col_cum, col_scree = st.columns(2)
    with col_cum:
        st.markdown("**Cumulative Explained Variance**")
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        ax3.plot(range(1, len(cumulative)+1), cumulative, color='#0ea5e9', linewidth=2.5)
        ax3.axhline(0.95, color='#10b981', linestyle='--', label='Ambang Batas 95%')
        ax3.fill_between(range(1, len(cumulative)+1), cumulative, alpha=0.1, color='#0ea5e9')
        ax3.set_xlabel("Jumlah Principal Components (k)")
        ax3.set_ylabel("Rasio Variansi Kumulatif")
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        ax3.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_scree:
        st.markdown("**Scree Plot (Distribusi Nilai Eigen)**")
        n_scree = min(30, len(eigenvalues_all))
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        ax4.plot(range(1, n_scree+1), eigenvalues_all[:n_scree], marker='o', color='#f59e0b', linewidth=2, markersize=6)
        ax4.set_xlabel("Indeks Komponen Utama")
        ax4.set_ylabel("Magnitudo Eigenvalue")
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)


# PROSES KOMPRESI (Cache)

@st.cache_data(show_spinner=False)
def jalankan_kompresi(gray_bytes: bytes, k_values_tuple: tuple):
    X_g = np.array(Image.open(io.BytesIO(gray_bytes)).convert("L"), dtype=np.float32)
    k_sample = min(200, X_g.shape[1])
    _, eig_all = pca_compress(X_g, k_sample)
    ev_ratio_all = eig_all / np.sum(eig_all)

    hasil, metrik = [], []
    for k in k_values_tuple:
        k_eff = min(k, X_g.shape[1])
        recon, _ = pca_compress(X_g, k_eff)
        recon8 = recon.astype(np.uint8)
        mse, psnr_val, ssim_val = hitung_metrik(X_g, recon)
        ev_cum = float(np.sum(ev_ratio_all[:k_eff])) * 100
        b_recon = img_to_bytes(recon8, "JPEG")
        uk_kb = ukuran_bytes(b_recon)

        hasil.append(recon8)
        metrik.append({
            "k": k_eff, "ev": ev_cum, "mse": mse, "psnr": psnr_val,
            "ssim": ssim_val, "ukuran_kb": uk_kb, "bytes": b_recon,
        })
    return hasil, metrik, X_g, eig_all

with st.spinner("Memproses rekonstruksi matriks..."):
    hasil_gray, metrik_list, X_gray_cached, eig_all_cached = jalankan_kompresi(
        bytes_asli, tuple(k_values)
    )
    ukuran_awal_kb_ref = ukuran_awal_kb


# TAB 3 – HASIL KOMPRESI

with tab3:
    st.markdown('<p class="section-header">Metrik Evaluasi Kompresi</p>', unsafe_allow_html=True)

    rows = []
    for m in metrik_list:
        penghematan = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
        cr = ukuran_awal_kb_ref / m["ukuran_kb"] if m["ukuran_kb"] > 0 else 0
        rows.append({
            "Dimensi (k)": m["k"],
            "Expl. Variance (%)": m['ev'],
            "MSE": m['mse'],
            "PSNR (dB)": m['psnr'],
            "SSIM": m['ssim'],
            "Ukuran (KB)": m['ukuran_kb'],
            "Rasio Kompresi": f"{cr:.2f}x",
            "Reduksi Memori (%)": penghematan,
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "Expl. Variance (%)": "{:.2f}",
            "MSE": "{:.2f}",
            "PSNR (dB)": "{:.2f}",
            "SSIM": "{:.4f}",
            "Ukuran (KB)": "{:.2f}",
            "Reduksi Memori (%)": "{:.1f}%"
        }).background_gradient(cmap='Blues', subset=['SSIM', 'Reduksi Memori (%)']),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>**Perbandingan Visual Rekonstruksi**", unsafe_allow_html=True)
    
    chunk_size = 4
    groups = [k_values[i:i+chunk_size] for i in range(0, len(k_values), chunk_size)]
    
    for grp in groups:
        cols_g = st.columns(len(grp) + 1)
        with cols_g[0]:
            st.image(X_gray_cached.astype(np.uint8), use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-size:0.9rem;'><b>Citra Asli</b><br>{ukuran_awal_kb_ref:.1f} KB</div>", unsafe_allow_html=True)
        for j, kv in enumerate(grp):
            idx = k_values.index(kv)
            m = metrik_list[idx]
            with cols_g[j+1]:
                st.image(hasil_gray[idx], use_container_width=True)
                st.markdown(f"<div style='text-align:center; font-size:0.9rem;'><b>k = {kv}</b><br>SSIM: {m['ssim']:.3f}</div>", unsafe_allow_html=True)


# TAB 4 – EDA SETELAH KOMPRESI

with tab4:
    st.markdown('<p class="section-header">Analisis Spasial & Distribusi Error</p>', unsafe_allow_html=True)

    n_eda = min(k_eda_count, len(k_values))
    step = max(1, len(k_values) // n_eda)
    idx_eda = list(dict.fromkeys([0] + list(range(0, len(k_values), step)) + [len(k_values)-1]))[:n_eda]

    for idx in idx_eda:
        k_cur = metrik_list[idx]["k"]
        m = metrik_list[idx]
        img_rec = hasil_gray[idx]

        st.markdown(f"**Konfigurasi Parameter: *k* = {k_cur}**")
        
        fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4.5))

        # Citra Rekonstruksi
        axes6[0].imshow(img_rec, cmap='gray')
        axes6[0].set_title(f"Rekonstruksi (PSNR: {m['psnr']:.2f} dB)", pad=10)
        axes6[0].axis('off')

        # Error Map
        err = np.abs(X_gray_cached - img_rec.astype(np.float32))
        im6 = axes6[1].imshow(err, cmap='magma')
        axes6[1].set_title(f"Peta Kesalahan (MSE: {m['mse']:.2f})", pad=10)
        axes6[1].axis('off')
        plt.colorbar(im6, ax=axes6[1], fraction=0.046, pad=0.04)

        # Histogram Overlay
        axes6[2].hist(X_gray_cached.flatten(), bins=128, alpha=0.5, color='#94a3b8', label='Original')
        axes6[2].hist(img_rec.flatten().astype(float), bins=128, alpha=0.6, color='#3b82f6', label=f'Rekonstruksi')
        axes6[2].set_title("Pergeseran Distribusi Intensitas", pad=10)
        axes6[2].spines['top'].set_visible(False); axes6[2].spines['right'].set_visible(False)
        axes6[2].legend(frameon=False)

        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)
        st.markdown("<hr style='border-color: #f1f5f9; margin: 2rem 0;'>", unsafe_allow_html=True)


# TAB 5 – DOWNLOAD

with tab5:
    st.markdown('<p class="section-header">Ekspor Hasil Kompresi</p>', unsafe_allow_html=True)

    # Identifikasi titik optimal secara heuristik
    best_idx = max(range(len(metrik_list)), key=lambda i: metrik_list[i]["ssim"] * 0.6 - (metrik_list[i]["ukuran_kb"] / ukuran_awal_kb_ref) * 0.4)
    bm = metrik_list[best_idx]
    
    st.markdown(f"""
    <div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 1.2rem; border-radius: 6px; margin-bottom: 2rem;">
        <h4 style="color: #166534; margin-top: 0;">Titik Optimal Direkomendasikan: k = {bm['k']}</h4>
        <p style="color: #15803d; margin-bottom: 0;">Konfigurasi ini memberikan keseimbangan terbaik antara retensi kualitas spasial (SSIM: {bm['ssim']:.4f}) dan efisiensi memori (Reduksi: {((ukuran_awal_kb_ref - bm['ukuran_kb']) / ukuran_awal_kb_ref) * 100:.1f}%).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Unduh Spesifik Berdasarkan *k***")
    dl_cols = st.columns(min(len(k_values), 4))
    for i, (kv, m) in enumerate(zip(k_values, metrik_list)):
        with dl_cols[i % 4]:
            penghematan = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
            st.download_button(
                label       = f"📥 Unduh k={kv} (-{penghematan:.0f}%)",
                data        = m["bytes"],
                file_name   = f"PCA_compressed_k{kv}.jpg",
                mime        = "image/jpeg",
                key         = f"dl_{kv}",
                use_container_width=True
            )

    st.markdown("<hr style='border-color: #f1f5f9; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("**Ekspor Komprehensif (Format ZIP)**")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for kv, m in zip(k_values, metrik_list):
            zf.writestr(f"PCA_compressed_k{kv}.jpg", m["bytes"])

        report_lines = [
            "LAPORAN ANALISIS KOMPRESI PCA",
            "=" * 70,
            f"Dimensi Citra  : {gambar.size[0]} x {gambar.size[1]} piksel",
            f"Ukuran Awal    : {ukuran_awal_kb_ref:.2f} KB",
            "",
            f"{'k':>5} | {'Expl.Var':>9} | {'MSE':>8} | {'PSNR':>9} | {'SSIM':>6} | {'Ukuran':>8} | {'Reduksi':>8}",
            "-" * 70,
        ]
        for m in metrik_list:
            ph = ((ukuran_awal_kb_ref - m["ukuran_kb"]) / ukuran_awal_kb_ref) * 100
            report_lines.append(
                f"{m['k']:>5} | {m['ev']:>8.2f}% | {m['mse']:>8.2f} | {m['psnr']:>8.2f} | {m['ssim']:>6.4f} | {m['ukuran_kb']:>6.2f}KB | {ph:>7.1f}%"
            )
        zf.writestr("laporan_analisis.txt", "\n".join(report_lines))

    zip_buf.seek(0)
    st.download_button(
        label       = "📦 Unduh Seluruh Hasil & Laporan (.zip)",
        data        = zip_buf.getvalue(),
        file_name   = "PCA_Analysis_Export.zip",
        mime        = "application/zip",
        key         = "dl_zip",
    )

st.markdown("<hr style='border-color: #f1f5f9; margin-top: 4rem;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "Sistem Analisis Kompresi Citra PCA & EDA"
    "</p>",
    unsafe_allow_html=True,
)
