import streamlit as st
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import zipfile
import pandas as pd

st.set_page_config(
    page_title="PCA Image Compressor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT       = "#7c3aed"        # Electric Violet
ACCENT_LIGHT = "#a78bfa"        # Soft Violet
ACCENT_GLOW  = "rgba(124,58,237,0.35)"
BG_BASE      = "#09090b"        # True Obsidian
BG_SURFACE   = "#111113"
BG_CARD      = "rgba(255,255,255,0.04)"
BORDER       = "rgba(255,255,255,0.08)"
TEXT_PRIMARY = "#f4f4f5"
TEXT_MUTED   = "#71717a"
SUCCESS      = "#22c55e"
WARNING      = "#f59e0b"
DANGER       = "#ef4444"

PLOT_BG      = "#0d0d10"
PLOT_GRID    = "rgba(255,255,255,0.06)"
PLOT_TEXT    = "#a1a1aa"

CSS = f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: {BG_BASE} !important;
    color: {TEXT_PRIMARY} !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1280px !important;
}}


::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT}; border-radius: 99px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {BG_SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1.2rem !important;
}}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {{
    color: {TEXT_MUTED} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}

/* ── Sidebar widgets ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER} !important;
    gap: 0.25rem;
    padding-bottom: 0 !important;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: {TEXT_MUTED} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em;
}}
[data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY} !important;
    background: rgba(255,255,255,0.04) !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    color: {ACCENT_LIGHT} !important;
    background: rgba(124,58,237,0.12) !important;
    border-bottom: 2px solid {ACCENT_LIGHT} !important;
}}
[data-baseweb="tab-highlight"] {{ display: none !important; }}
[data-baseweb="tab-border"] {{ display: none !important; }}

/* ── File uploader dropzone ── */
[data-testid="stFileUploader"] section {{
    background: rgba(124,58,237,0.05) !important;
    border: 1.5px dashed rgba(124,58,237,0.4) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {ACCENT_LIGHT} !important;
    background: rgba(124,58,237,0.10) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px {ACCENT_GLOW};
}}
[data-testid="stFileUploader"] section svg {{ display: none !important; }}
[data-testid="stFileUploader"] section > div > p:first-child {{
    font-size: 0.85rem !important;
    color: {ACCENT_LIGHT} !important;
    font-weight: 600 !important;
}}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] button {{
    background: rgba(124,58,237,0.12) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    color: {ACCENT_LIGHT} !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.9rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.01em;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: rgba(124,58,237,0.25) !important;
    border-color: {ACCENT_LIGHT} !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px {ACCENT_GLOW} !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] {{
    color: {ACCENT_LIGHT} !important;
}}

/* ── Text input ── */
[data-testid="stTextInput"] input {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.2s ease !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {ACCENT_LIGHT} !important;
    box-shadow: 0 0 0 2px {ACCENT_GLOW} !important;
    outline: none !important;
}}
[data-testid="stTextInput"] input::placeholder {{
    color: {TEXT_MUTED} !important;
    font-size: 0.8rem !important;
}}

/* ── Slider ── */
[data-baseweb="slider"] [data-testid="stSlider"] div[role="slider"] {{
    background: {ACCENT} !important;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 1.5rem 0 !important; }}

/* ── Metric cards hover (custom) ── */
.pca-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}}
.pca-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_LIGHT});
    opacity: 0;
    transition: opacity 0.3s ease;
}}
.pca-card:hover {{
    transform: translateY(-4px);
    border-color: rgba(124,58,237,0.4);
    box-shadow: 0 12px 40px {ACCENT_GLOW};
}}
.pca-card:hover::before {{ opacity: 1; }}
.pca-card .card-label {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_MUTED};
    margin-bottom: 0.35rem;
}}
.pca-card .card-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.pca-card .card-sub {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-top: 0.3rem;
}}
.card-accent {{ color: {ACCENT_LIGHT} !important; }}

/* ── Section header ── */
.sec-header {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXT_MUTED};
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}}
.sec-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

/* ── Badge ── */
.badge {{
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    border: 1px solid;
}}
.badge-violet {{
    color: {ACCENT_LIGHT};
    border-color: rgba(124,58,237,0.4);
    background: rgba(124,58,237,0.12);
}}
.badge-green {{
    color: {SUCCESS};
    border-color: rgba(34,197,94,0.3);
    background: rgba(34,197,94,0.08);
}}
.badge-amber {{
    color: {WARNING};
    border-color: rgba(245,158,11,0.3);
    background: rgba(245,158,11,0.08);
}}

/* ── Optimal card ── */
.optimal-card {{
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(167,139,250,0.08));
    border: 1px solid rgba(124,58,237,0.45);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
}}
.optimal-card::after {{
    content: '✦';
    position: absolute;
    right: 1.5rem;
    top: 1.2rem;
    font-size: 2.5rem;
    color: rgba(124,58,237,0.15);
}}

/* ── Table ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] table {{
    background: {BG_CARD} !important;
}}
[data-testid="stDataFrame"] th {{
    background: rgba(124,58,237,0.12) !important;
    color: {ACCENT_LIGHT} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600 !important;
}}
[data-testid="stDataFrame"] td {{
    font-size: 0.82rem !important;
    color: {TEXT_PRIMARY} !important;
    border-color: {BORDER} !important;
}}

/* ── Spinner overlay ── */
@keyframes pca-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.95); }}
}}
.pca-loading {{
    animation: pca-pulse 1.5s ease-in-out infinite;
    color: {ACCENT_LIGHT};
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}}

/* ── Image frame ── */
.img-frame {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.3s ease;
    background: {BG_SURFACE};
}}
.img-frame:hover {{
    border-color: rgba(124,58,237,0.4);
    box-shadow: 0 8px 32px {ACCENT_GLOW};
}}
.img-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {TEXT_MUTED};
    margin-top: 0.5rem;
    text-align: center;
}}
.img-meta {{
    font-size: 0.7rem;
    color: {TEXT_MUTED};
    text-align: center;
    margin-top: 0.1rem;
}}

/* ── Alert/info boxes override ── */
[data-testid="stAlert"] {{
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important;
    color: {TEXT_PRIMARY} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def setup_plt():
    plt.rcParams.update({
        "figure.facecolor"    : PLOT_BG,
        "axes.facecolor"      : PLOT_BG,
        "axes.edgecolor"      : "none",
        "axes.labelcolor"     : PLOT_TEXT,
        "axes.titlecolor"     : TEXT_PRIMARY,
        "axes.titlesize"      : 11,
        "axes.titleweight"    : "600",
        "axes.labelsize"      : 9,
        "axes.grid"           : True,
        "grid.color"          : PLOT_GRID,
        "grid.linewidth"      : 0.7,
        "xtick.color"         : PLOT_TEXT,
        "ytick.color"         : PLOT_TEXT,
        "xtick.labelsize"     : 8,
        "ytick.labelsize"     : 8,
        "legend.facecolor"    : "#18181b",
        "legend.edgecolor"    : BORDER,
        "legend.fontsize"     : 8,
        "legend.labelcolor"   : TEXT_PRIMARY,
        "lines.linewidth"     : 2.2,
        "savefig.facecolor"   : PLOT_BG,
        "savefig.transparent" : False,
        "font.family"         : "sans-serif",
    })

setup_plt()

def pca_compress(channel: np.ndarray, k: int):
    mean          = np.mean(channel, axis=0)
    Xc            = channel - mean
    cov_matrix    = np.cov(Xc, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx    = np.argsort(eigenvalues)[::-1]
    eigenvalues   = eigenvalues[sorted_idx]
    eigenvectors  = eigenvectors[:, sorted_idx]
    W             = eigenvectors[:, :k]
    Z             = np.dot(Xc, W)
    reconstructed = np.dot(Z, W.T) + mean
    return np.clip(reconstructed, 0, 255), eigenvalues


def hitung_metrik(original, rekon):
    mse      = np.mean((original - rekon) ** 2)
    psnr     = float('inf') if mse == 0 else 10 * np.log10((255**2) / mse)
    ssim_val = ssim(original.astype(np.uint8), rekon.astype(np.uint8), data_range=255)
    return mse, psnr, ssim_val


def to_bytes(arr, fmt="JPEG"):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


def kb(b): return len(b) / 1024


def card(label, value, sub="", accent=False):
    val_cls = 'card-value card-accent' if accent else 'card-value'
    return f"""
    <div class="pca-card">
        <div class="card-label">{label}</div>
        <div class="{val_cls}">{value}</div>
        {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def sec(icon, title):
    st.markdown(f'<div class="sec-header">{icon}&nbsp; {title}</div>', unsafe_allow_html=True)


def badge(text, kind="violet"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def quality_badge(psnr):
    if psnr >= 40:   return badge("Sangat Baik", "green")
    elif psnr >= 30: return badge("Baik", "violet")
    else:            return badge("Sedang", "amber")


with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <div style="font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; color:{TEXT_PRIMARY}">
            ✦ PCA Compressor
        </div>
        <div style="font-size:0.72rem; color:{TEXT_MUTED}; margin-top:0.2rem; letter-spacing:0.05em; text-transform:uppercase">
            Eigenimage · EDA · Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};margin-bottom:0.4rem">Upload Citra</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", type=["jpg","jpeg","png","bmp","webp"],
        help="JPG · PNG · BMP · WebP"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};margin-bottom:0.4rem">Nilai K — Input Manual</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.68rem;color:{TEXT_MUTED};margin-bottom:0.5rem;line-height:1.5">Ketik beberapa nilai k dipisah koma.<br>Contoh: <span style="color:{ACCENT_LIGHT};font-weight:600">5, 10, 20, 50, 100</span></div>', unsafe_allow_html=True)

    k_input_raw = st.text_input(
        "", value="5, 10, 20, 50, 100",
        placeholder="Contoh: 5, 10, 20, 50, 100",
        key="k_input",
    )

    # Parse & validasi input
    k_parse_error = None
    k_preset      = []
    try:
        parts = [p.strip() for p in k_input_raw.split(",") if p.strip()]
        if not parts:
            k_parse_error = "Masukkan minimal satu nilai k."
        else:
            parsed = []
            for p in parts:
                v = int(p)
                if v < 1:
                    k_parse_error = f"Nilai k harus ≥ 1 (ditemukan: {v})."
                    break
                if v > 2000:
                    k_parse_error = f"Nilai k terlalu besar (maks 2000, ditemukan: {v})."
                    break
                parsed.append(v)
            if not k_parse_error:
                k_preset = sorted(set(parsed))
    except ValueError:
        k_parse_error = "Format tidak valid. Gunakan angka bulat dipisah koma."

    if k_parse_error:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#ef4444;background:rgba(239,68,68,0.08);'
            f'border:1px solid rgba(239,68,68,0.25);border-radius:8px;padding:0.5rem 0.7rem;margin-top:0.3rem">'
            f'⚠ {k_parse_error}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.7rem;color:#22c55e;background:rgba(34,197,94,0.08);'
            f'border:1px solid rgba(34,197,94,0.2);border-radius:8px;padding:0.4rem 0.7rem;margin-top:0.3rem">'
            f'✓ {len(k_preset)} nilai k: {", ".join(str(x) for x in k_preset)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};margin-bottom:0.4rem">EDA Detail</div>', unsafe_allow_html=True)
    k_eda_count = st.slider("", 1, min(5, max(len(k_preset) if k_preset else 1, 1)), min(3, max(len(k_preset) if k_preset else 1, 1)))

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{TEXT_MUTED}; line-height:1.7">
        Kompresi citra berbasis <span style="color:{ACCENT_LIGHT}; font-weight:600">PCA</span>
        menggunakan <span style="color:{ACCENT_LIGHT}; font-weight:600">Eigenvalue</span> &amp;
        <span style="color:{ACCENT_LIGHT}; font-weight:600">Eigenvector</span>.<br><br>
        Metrik evaluasi: MSE · PSNR · SSIM · CR
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown(f"""
    <div style="min-height:80vh; display:flex; flex-direction:column;
                align-items:center; justify-content:center; text-align:center; padding:4rem 2rem;">

        <div style="font-size:3.8rem; margin-bottom:1rem; filter:drop-shadow(0 0 40px {ACCENT_GLOW})">✦</div>

        <h1 style="font-size:3rem; font-weight:800; letter-spacing:-0.04em;
                   background:linear-gradient(135deg, {TEXT_PRIMARY} 0%, {ACCENT_LIGHT} 100%);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin:0 0 0.6rem 0; line-height:1.1">
            PCA Image Compressor
        </h1>

        <p style="font-size:1rem; color:{TEXT_MUTED}; max-width:480px; line-height:1.7; margin-bottom:2.5rem">
            Kompresi citra berbasis <strong style="color:{ACCENT_LIGHT}">Principal Component Analysis</strong>
            menggunakan Eigenvalue &amp; Eigenvector — dilengkapi analisis EDA mendalam
            sebelum dan sesudah kompresi.
        </p>

        <div style="display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:center; margin-bottom:3rem">
            {badge("Eigenvalue Analysis","violet")}
            {badge("PSNR · SSIM · MSE","violet")}
            {badge("EDA Visual","violet")}
            {badge("Batch Download","violet")}
        </div>

        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; max-width:600px; width:100%; margin-bottom:3rem">
            <div class="pca-card" style="text-align:left">
                <div style="font-size:1.3rem;margin-bottom:0.4rem">📊</div>
                <div style="font-size:0.8rem;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:0.2rem">EDA Lengkap</div>
                <div style="font-size:0.7rem;color:{TEXT_MUTED}">Histogram, statistik piksel, dan analisis distribusi intensitas.</div>
            </div>
            <div class="pca-card" style="text-align:left">
                <div style="font-size:1.3rem;margin-bottom:0.4rem">📈</div>
                <div style="font-size:0.8rem;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:0.2rem">Scree Plot</div>
                <div style="font-size:0.7rem;color:{TEXT_MUTED}">Visualisasi eigenvalue dan cumulative explained variance.</div>
            </div>
            <div class="pca-card" style="text-align:left">
                <div style="font-size:1.3rem;margin-bottom:0.4rem">🗜️</div>
                <div style="font-size:0.8rem;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:0.2rem">Multi-k Compress</div>
                <div style="font-size:0.7rem;color:{TEXT_MUTED}">Uji berbagai nilai k dan bandingkan hasilnya secara visual.</div>
            </div>
        </div>

        <div style="font-size:0.78rem;color:{TEXT_MUTED}">
            ↑ Upload citra di sidebar kiri untuk memulai
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if k_parse_error or not k_preset:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.8rem;padding:1rem 1.3rem;
                background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);
                border-radius:10px;margin-top:1rem">
        <span style="font-size:1.2rem">⚠️</span>
        <span style="font-size:0.85rem;color:#fca5a5">
            {k_parse_error or "Masukkan minimal satu nilai k yang valid di sidebar."}
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

gambar         = Image.open(uploaded_file)
mode_citra     = "Grayscale" if gambar.mode == "L" else "Berwarna"
X_rgb          = np.array(gambar.convert("RGB"),  dtype=np.float32)
X_gray         = np.array(gambar.convert("L"),    dtype=np.float32)
bytes_asli     = uploaded_file.getvalue()
ukuran_awal_kb = kb(bytes_asli)
max_k          = X_gray.shape[1]
k_values       = sorted(set(min(k, max_k) for k in k_preset))


fn = uploaded_file.name
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid {BORDER}">
    <div>
        <h2 style="font-size:1.5rem; font-weight:800; letter-spacing:-0.03em;
                   color:{TEXT_PRIMARY}; margin:0 0 0.2rem 0">✦ PCA Image Compressor</h2>
        <div style="font-size:0.75rem; color:{TEXT_MUTED}">
            {fn} &nbsp;·&nbsp; {gambar.size[0]} × {gambar.size[1]} px
            &nbsp;·&nbsp; {ukuran_awal_kb:.1f} KB
            &nbsp;·&nbsp; {mode_citra}
        </div>
    </div>
    <div style="display:flex;gap:0.5rem;align-items:center">
        {badge(f"{len(k_values)} nilai k","violet")}
        {badge(mode_citra.lower(),"green")}
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  EDA Awal  ",
    "  📈  Eigenvalue  ",
    "  🗜️  Kompresi  ",
    "  🔍  EDA Pasca  ",
    "  ⬇️  Export  ",
])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    sec("📊", "Overview Citra")

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card("Ukuran", f"{gambar.size[0]}×{gambar.size[1]}", "piksel"), unsafe_allow_html=True)
    with c2: st.markdown(card("File Asli", f"{ukuran_awal_kb:.1f} KB", "sebelum kompresi"), unsafe_allow_html=True)
    with c3: st.markdown(card("Mean Piksel", f"{X_gray.mean():.1f}", "intensitas rata-rata", accent=True), unsafe_allow_html=True)
    with c4: st.markdown(card("Std Dev", f"{X_gray.std():.1f}", "variasi intensitas", accent=True), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_img, col_hist = st.columns([1.2, 1])

    with col_img:
        sec("🖼️", "Citra Asli vs Grayscale")
        fig, axes = plt.subplots(1,2, figsize=(10,4.2), tight_layout=True)
        axes[0].imshow(X_rgb.astype(np.uint8))
        axes[0].set_title("RGB Original", pad=10)
        axes[0].axis('off')
        axes[1].imshow(X_gray, cmap='gray')
        axes[1].set_title("Grayscale Channel", pad=10)
        axes[1].axis('off')
        for sp in axes[0].spines.values(): sp.set_visible(False)
        for sp in axes[1].spines.values(): sp.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_hist:
        sec("📉", "Distribusi Intensitas Piksel")

        mn = X_gray.mean()
        if mn < 85:   kecerahan, b_kec = "Gelap",  "amber"
        elif mn > 170: kecerahan, b_kec = "Terang", "green"
        else:          kecerahan, b_kec = "Normal", "violet"

        sd = X_gray.std()
        if sd < 40:   kontras, b_kon = "Rendah", "amber"
        elif sd > 80: kontras, b_kon = "Tinggi", "green"
        else:          kontras, b_kon = "Sedang", "violet"

        st.markdown(
            f"Kecerahan&nbsp;{badge(kecerahan, b_kec)}&nbsp;&nbsp;"
            f"Kontras&nbsp;{badge(kontras, b_kon)}",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6,3.8), tight_layout=True)
        ax2.fill_between(range(256),
            [np.sum(X_gray.flatten() == i) for i in range(256)],
            alpha=0.25, color=ACCENT_LIGHT)
        n, bins, patches = ax2.hist(X_gray.flatten(), bins=64,
                                    color=ACCENT, edgecolor="none", alpha=0.75)
        ax2.axvline(mn, color="#f59e0b", linestyle='--', linewidth=1.5,
                    label=f"Mean = {mn:.1f}")
        ax2.set_xlabel("Intensitas (0–255)")
        ax2.set_ylabel("Frekuensi")
        ax2.set_title("Histogram Piksel — Grayscale", pad=10)
        ax2.legend()
        for sp in ['top','right','left','bottom']: ax2.spines[sp].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    sec("📈", "Analisis Eigenvalue & Explained Variance")

    with st.spinner("Menghitung dekomposisi eigen…"):
        k_sample = min(200, X_gray.shape[1])
        _, eig_all = pca_compress(X_gray, k_sample)

    ev_ratio   = eig_all / np.sum(eig_all)
    cumulative = np.cumsum(ev_ratio)

    # Top 5 cards
    cols_ev = st.columns(5)
    for i in range(min(5, len(eig_all))):
        with cols_ev[i]:
            st.markdown(card(f"PC {i+1}", f"{eig_all[i]:,.0f}",
                             f"{ev_ratio[i]*100:.1f}% variance", accent=(i==0)),
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Threshold table
    sec("🎯", "Threshold Explained Variance")
    th_cols = st.columns(4)
    for idx_t, thr in enumerate([0.80, 0.90, 0.95, 0.99]):
        k_n = int(np.argmax(cumulative >= thr)) + 1
        with th_cols[idx_t]:
            st.markdown(card(
                f"{int(thr*100)}% Variance",
                f"k = {k_n}",
                "komponen dibutuhkan",
                accent=(thr == 0.95)
            ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cum, col_scree = st.columns(2)
    with col_cum:
        sec("📊", "Cumulative Explained Variance")
        fig3, ax3 = plt.subplots(figsize=(6,3.8), tight_layout=True)
        ax3.plot(range(1, len(cumulative)+1), cumulative,
                 color=ACCENT_LIGHT, linewidth=2.5)
        ax3.fill_between(range(1, len(cumulative)+1), cumulative,
                         alpha=0.12, color=ACCENT)
        for thr, col, lbl in [(0.90,"#f59e0b","90%"),
                               (0.95,"#a78bfa","95%"),
                               (0.99,"#22c55e","99%")]:
            ax3.axhline(thr, color=col, linestyle='--', linewidth=1.3, label=lbl)
        ax3.set_xlabel("Komponen Utama (k)")
        ax3.set_ylabel("Cumulative Explained Variance")
        ax3.set_title("Cumulative Explained Variance", pad=10)
        ax3.legend(); ax3.set_ylim(0, 1.05)
        for sp in ['top','right']: ax3.spines[sp].set_visible(False)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with col_scree:
        sec("📉", "Scree Plot")
        n_sc = min(30, len(eig_all))
        fig4, ax4 = plt.subplots(figsize=(6,3.8), tight_layout=True)
        ax4.plot(range(1, n_sc+1), eig_all[:n_sc],
                 marker='o', color=ACCENT_LIGHT, linewidth=2.2,
                 markerfacecolor=ACCENT, markersize=5.5)
        ax4.fill_between(range(1, n_sc+1), eig_all[:n_sc],
                         alpha=0.1, color=ACCENT)
        ax4.set_xlabel("Komponen (k)")
        ax4.set_ylabel("Eigenvalue")
        ax4.set_title(f"Scree Plot — {n_sc} Eigenvalue Terbesar", pad=10)
        for sp in ['top','right']: ax4.spines[sp].set_visible(False)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)


@st.cache_data(show_spinner=False)
def jalankan_kompresi(gray_bytes, k_tuple):
    Xg = np.array(Image.open(io.BytesIO(gray_bytes)).convert("L"), dtype=np.float32)
    k_s = min(200, Xg.shape[1])
    _, eig_a = pca_compress(Xg, k_s)
    ev_r = eig_a / np.sum(eig_a)
    hasil, metrik = [], []
    for k in k_tuple:
        ke = min(k, Xg.shape[1])
        r, _ = pca_compress(Xg, ke)
        r8 = r.astype(np.uint8)
        mse, psnr_v, ssim_v = hitung_metrik(Xg, r)
        ev_c = float(np.sum(ev_r[:ke])) * 100
        b = to_bytes(r8, "JPEG")
        hasil.append(r8)
        metrik.append({"k":ke,"ev":ev_c,"mse":mse,"psnr":psnr_v,
                       "ssim":ssim_v,"ukuran_kb":kb(b),"bytes":b})
    return hasil, metrik, Xg, eig_a

with st.spinner("Memproses kompresi untuk semua nilai k…"):
    hasil_gray, metrik_list, Xg_c, eig_c = jalankan_kompresi(
        uploaded_file.getvalue(), tuple(k_values)
    )

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🗜️", "Tabel Evaluasi Kompresi")

    rows = []
    for m in metrik_list:
        ph = ((ukuran_awal_kb - m["ukuran_kb"]) / ukuran_awal_kb) * 100
        cr = ukuran_awal_kb / m["ukuran_kb"] if m["ukuran_kb"] > 0 else 0
        rows.append({
            "k"              : m["k"],
            "Expl. Var (%)"  : round(m["ev"],2),
            "MSE"            : round(m["mse"],2),
            "PSNR (dB)"      : round(m["psnr"],2),
            "SSIM"           : round(m["ssim"],4),
            "Ukuran (KB)"    : round(m["ukuran_kb"],2),
            "Hemat (%)"      : round(ph,1),
            "CR (×)"         : round(cr,2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("🖼️", "Komparasi Visual")

    # Grid 4 per baris
    chunk = 4
    groups = [k_values[i:i+chunk] for i in range(0, len(k_values), chunk)]
    for grp in groups:
        cols_g = st.columns(len(grp)+1)
        with cols_g[0]:
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(Xg_c.astype(np.uint8), use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="img-label">Original</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="img-meta">{ukuran_awal_kb:.1f} KB</div>', unsafe_allow_html=True)
        for j, kv in enumerate(grp):
            idx = k_values.index(kv)
            m   = metrik_list[idx]
            with cols_g[j+1]:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(hasil_gray[idx], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                ph2 = ((ukuran_awal_kb - m["ukuran_kb"]) / ukuran_awal_kb) * 100
                st.markdown(f'<div class="img-label">k = {kv}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="img-meta">'
                    f'PSNR {m["psnr"]:.1f} dB &nbsp;·&nbsp; SSIM {m["ssim"]:.3f}<br>'
                    f'{m["ukuran_kb"]:.1f} KB &nbsp;·&nbsp; -{ph2:.0f}%'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("📈", "Grafik Metrik")

    ks      = [m["k"]    for m in metrik_list]
    mses    = [m["mse"]  for m in metrik_list]
    psnrs   = [m["psnr"] for m in metrik_list]
    ssims   = [m["ssim"] for m in metrik_list]
    ukurans = [m["ukuran_kb"] for m in metrik_list]

    fig5, ax5 = plt.subplots(2,2, figsize=(13,8), tight_layout=True)

    def _plot(ax, y, color, title, ylabel, refs=None):
        ax.plot(ks, y, marker='o', color=color, linewidth=2.2,
                markerfacecolor=BG_BASE, markeredgecolor=color, markersize=6)
        ax.fill_between(ks, y, alpha=0.08, color=color)
        if refs:
            for val, lbl, c in refs:
                ax.axhline(val, color=c, linestyle='--', linewidth=1.2, label=lbl, alpha=0.8)
            ax.legend()
        ax.set_title(title, pad=10)
        ax.set_xlabel("k (komponen utama)")
        ax.set_ylabel(ylabel)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)

    _plot(ax5[0,0], mses,    "#ef4444", "MSE vs k",             "MSE")
    _plot(ax5[0,1], psnrs,   ACCENT_LIGHT, "PSNR vs k",         "PSNR (dB)",
          [(30,"30 dB — Baik","#f59e0b"),(40,"40 dB — Sangat Baik","#22c55e")])
    _plot(ax5[1,0], ssims,   "#22c55e", "SSIM vs k",            "SSIM",
          [(0.90,"SSIM = 0.90","#f59e0b")])
    ax5[1,0].set_ylim(0, 1.05)
    _plot(ax5[1,1], ukurans, "#f59e0b", "Ukuran File vs k",     "Ukuran (KB)")
    ax5[1,1].axhline(ukuran_awal_kb, color=TEXT_MUTED, linestyle='--',
                     linewidth=1.2, label=f"Asli = {ukuran_awal_kb:.1f} KB", alpha=0.6)
    ax5[1,1].legend()

    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)


with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🔍", "EDA Setelah Kompresi — Detail per k")

    n_eda = min(k_eda_count, len(k_values))
    step  = max(1, len(k_values) // n_eda)
    idx_eda = list(dict.fromkeys(
        [0] + list(range(0, len(k_values), step)) + [len(k_values)-1]
    ))[:n_eda]

    for idx in idx_eda:
        m   = metrik_list[idx]
        k_c = m["k"]
        img = hasil_gray[idx]
        ph  = ((ukuran_awal_kb - m["ukuran_kb"]) / ukuran_awal_kb) * 100

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem">'
            f'<span style="font-size:1rem;font-weight:700;color:{TEXT_PRIMARY}">k = {k_c}</span>'
            f'&nbsp;{quality_badge(m["psnr"])}'
            f'&nbsp;<span style="font-size:0.72rem;color:{TEXT_MUTED}">·&nbsp; -{ph:.0f}% ukuran</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(card("MSE",  f'{m["mse"]:.1f}',  "error per piksel²"), unsafe_allow_html=True)
        with c2: st.markdown(card("PSNR", f'{m["psnr"]:.2f}', "dB — kualitas", accent=True), unsafe_allow_html=True)
        with c3: st.markdown(card("SSIM", f'{m["ssim"]:.4f}', "structural similarity", accent=True), unsafe_allow_html=True)
        with c4: st.markdown(card("Hemat", f'{ph:.1f}%',      f'{m["ukuran_kb"]:.1f} KB hasil'), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig6, axes6 = plt.subplots(1,4, figsize=(18,4.2), tight_layout=True)

        axes6[0].imshow(Xg_c.astype(np.uint8), cmap='gray')
        axes6[0].set_title("Original", pad=8); axes6[0].axis('off')

        axes6[1].imshow(img, cmap='gray')
        axes6[1].set_title(f"Rekonstruksi k={k_c}", pad=8); axes6[1].axis('off')

        err = np.abs(Xg_c - img.astype(np.float32))
        im6 = axes6[2].imshow(err, cmap='plasma')
        axes6[2].set_title(f"Error Map · MSE={m['mse']:.2f}", pad=8); axes6[2].axis('off')
        cbar = fig6.colorbar(im6, ax=axes6[2], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=PLOT_TEXT, labelsize=7)

        axes6[3].hist(Xg_c.flatten(), bins=64, alpha=0.65, color=ACCENT_LIGHT, label='Asli')
        axes6[3].hist(img.flatten().astype(float), bins=64, alpha=0.65, color="#f59e0b", label=f'k={k_c}')
        axes6[3].set_xlabel("Intensitas"); axes6[3].set_ylabel("Frekuensi")
        axes6[3].set_title("Histogram Perbandingan", pad=8)
        axes6[3].legend()
        for sp in ['top','right']: axes6[3].spines[sp].set_visible(False)

        st.pyplot(fig6, use_container_width=True)
        plt.close(fig6)
        st.markdown(f'<hr style="border-color:{BORDER};margin:1.5rem 0">', unsafe_allow_html=True)

with tab5:
    st.markdown("<br>", unsafe_allow_html=True)

    # Optimal Card
    best_i = max(range(len(metrik_list)),
                 key=lambda i: (metrik_list[i]["ssim"]*0.5
                                + (metrik_list[i]["psnr"]/50)*0.3
                                - (metrik_list[i]["ukuran_kb"]/ukuran_awal_kb)*0.2))
    bm  = metrik_list[best_i]
    bph = ((ukuran_awal_kb - bm["ukuran_kb"]) / ukuran_awal_kb) * 100
    bcr = ukuran_awal_kb / bm["ukuran_kb"]

    st.markdown(f"""
    <div class="optimal-card" style="margin-bottom:2rem">
        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.12em;color:{ACCENT_LIGHT};margin-bottom:0.8rem">
            ✦ Titik Optimal Direkomendasikan
        </div>
        <div style="font-size:2.4rem;font-weight:800;letter-spacing:-0.04em;
                    color:{TEXT_PRIMARY};margin-bottom:0.6rem">k = {bm["k"]}</div>
        <div style="display:flex;gap:2.5rem;flex-wrap:wrap">
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">Explained Var</div>
                <div style="font-size:1.1rem;font-weight:700;color:{ACCENT_LIGHT}">{bm["ev"]:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">PSNR</div>
                <div style="font-size:1.1rem;font-weight:700;color:{TEXT_PRIMARY}">{bm["psnr"]:.2f} dB</div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">SSIM</div>
                <div style="font-size:1.1rem;font-weight:700;color:{TEXT_PRIMARY}">{bm["ssim"]:.4f}</div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">Ukuran</div>
                <div style="font-size:1.1rem;font-weight:700;color:{SUCCESS}">{bm["ukuran_kb"]:.1f} KB</div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">Hemat</div>
                <div style="font-size:1.1rem;font-weight:700;color:{SUCCESS}">-{bph:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.65rem;color:{TEXT_MUTED};text-transform:uppercase;
                            letter-spacing:0.08em">Rasio Kompresi</div>
                <div style="font-size:1.1rem;font-weight:700;color:{TEXT_PRIMARY}">{bcr:.2f}×</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sec("⬇️", "Download Per Nilai k")

    dl_cols = st.columns(min(len(k_values), 4))
    for i, (kv, m) in enumerate(zip(k_values, metrik_list)):
        ph2 = ((ukuran_awal_kb - m["ukuran_kb"]) / ukuran_awal_kb) * 100
        with dl_cols[i % 4]:
            st.download_button(
                label     = f"k={kv} · {m['ukuran_kb']:.0f}KB · -{ph2:.0f}%",
                data      = m["bytes"],
                file_name = f"PCA_k{kv}.jpg",
                mime      = "image/jpeg",
                key       = f"dl_{kv}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    sec("📦", "Download Semua Sekaligus")

    # Build ZIP + laporan
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for kv, m in zip(k_values, metrik_list):
            zf.writestr(f"PCA_k{kv}.jpg", m["bytes"])
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║          LAPORAN KOMPRESI PCA                    ║",
            "╚══════════════════════════════════════════════════╝",
            f"File        : {uploaded_file.name}",
            f"Ukuran Asli : {ukuran_awal_kb:.2f} KB",
            f"Dimensi     : {gambar.size[0]} × {gambar.size[1]} px",
            "",
            f"{'k':>5} | {'Expl.Var':>9} | {'MSE':>8} | {'PSNR':>9} | {'SSIM':>7} | {'KB':>7} | {'Hemat':>6}",
            "─" * 67,
        ]
        for m in metrik_list:
            ph3 = ((ukuran_awal_kb - m["ukuran_kb"]) / ukuran_awal_kb) * 100
            lines.append(
                f"{m['k']:>5} | {m['ev']:>8.2f}% | {m['mse']:>8.2f} | "
                f"{m['psnr']:>8.2f} | {m['ssim']:>7.4f} | {m['ukuran_kb']:>6.1f} | {ph3:>5.1f}%"
            )
        lines += ["", f"★ Optimal: k={bm['k']} (PSNR={bm['psnr']:.2f} dB · SSIM={bm['ssim']:.4f})"]
        zf.writestr("laporan_kompresi.txt", "\n".join(lines))
    zip_buf.seek(0)

    st.download_button(
        label     = "📦 Download semua hasil (.zip) + laporan",
        data      = zip_buf.getvalue(),
        file_name = "PCA_compressed_all.zip",
        mime      = "application/zip",
        key       = "dl_zip",
    )

# ── Footer ──────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:3rem 0 1rem; color:{TEXT_MUTED};
            font-size:0.7rem; letter-spacing:0.06em; text-transform:uppercase">
    ✦ PCA Image Compressor &nbsp;·&nbsp; Eigenvalue &amp; Eigenvector &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
