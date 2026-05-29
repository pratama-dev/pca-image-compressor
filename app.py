import io
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False


st.set_page_config(
    page_title="PCA Image Compression Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_BASE = "#0b0f14"
BG_SURFACE = "#111823"
BG_PANEL = "rgba(255, 255, 255, 0.04)"
BORDER = "rgba(255, 255, 255, 0.10)"
TEXT_PRIMARY = "#f4f7fb"
TEXT_MUTED = "#c5d1e0"
CYAN = "#22d3ee"
PURPLE = "#a855f7"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            background: {BG_BASE} !important;
            color: {TEXT_PRIMARY} !important;
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        [data-testid="stSidebar"] {{
            background: {BG_SURFACE} !important;
            border-right: 1px solid {BORDER} !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }}

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p {{
            color: #e7eef9 !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 800 !important;
        }}

        [data-testid="stSidebar"] input::placeholder {{
            color: #9eb0c4 !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
            background: rgba(15, 23, 42, 0.88) !important;
            border: 1.4px dashed rgba(34, 211, 238, 0.28) !important;
            border-radius: 1rem !important;
            transition: all 0.25s ease;
        }}

        /* Hover uploader */
            [data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {{
            border-color: rgba(168, 85, 247, 0.55) !important;
            background: rgba(17, 24, 39, 0.96) !important;
        }}

        /* Tulisan uploader */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
            color: #dbe7f5 !important;
        }}

        /* Tombol browse/upload */
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
            background: linear-gradient(
                90deg,
                rgba(34, 211, 238, 0.14),
                rgba(168, 85, 247, 0.14)
            ) !important;

            border: 1px solid rgba(34, 211, 238, 0.22) !important;
            color: #f4f7fb !important;
            border-radius: 0.8rem !important;
            font-weight: 600 !important;
        }}

        /* Hover tombol */
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {{
            border-color: rgba(168, 85, 247, 0.55) !important;

            background: linear-gradient(
                90deg,
                rgba(34, 211, 238, 0.22),
                rgba(168, 85, 247, 0.22)
            ) !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="slider"] {{
            color: {TEXT_PRIMARY} !important;
        }}

        [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid {BORDER};
        }}

        [data-baseweb="tab"] {{
            color: {TEXT_MUTED};
            background: transparent;
            border-radius: 0.9rem 0.9rem 0 0;
            padding: 0.65rem 1rem;
            font-weight: 600;
        }}

        [aria-selected="true"][data-baseweb="tab"] {{
            color: {TEXT_PRIMARY};
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.13), rgba(168, 85, 247, 0.13));
            border-bottom: 2px solid {CYAN};
        }}

        .hero-title {{
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.05;
            margin: 0;
            background: linear-gradient(90deg, {CYAN}, {PURPLE});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hero-subtitle {{
            color: {TEXT_MUTED};
            margin-top: 0.35rem;
            font-size: 0.93rem;
            line-height: 1.6;
        }}

        .panel {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 1.1rem;
            padding: 1rem 1.1rem;
            position: relative;
            overflow: hidden;
        }}

        .panel::before {{
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 2px;
            background: linear-gradient(90deg, {CYAN}, {PURPLE});
            opacity: 0.95;
        }}

        .section-label {{
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }}

        .small-note {{
            color: {TEXT_MUTED};
            font-size: 0.86rem;
            line-height: 1.6;
        }}

        [data-testid="stMetric"] {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 1rem;
            padding: 0.8rem 0.9rem;
        }}

        [data-testid="stMetric"] label {{
            color: {TEXT_MUTED} !important;
        }}

        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY};
        }}

        [data-testid="stDownloadButton"] button {{
            width: 100%;
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.12), rgba(168, 85, 247, 0.12));
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(34, 211, 238, 0.25);
            border-radius: 0.85rem;
            transition: all 0.2s ease;
        }}

        [data-testid="stDownloadButton"] button:hover {{
            transform: translateY(-2px);
            border-color: rgba(168, 85, 247, 0.55);
            box-shadow: 0 10px 24px rgba(168, 85, 247, 0.12);
        }}

        .tech-box {{
            border: 1px solid {BORDER};
            border-radius: 1rem;
            padding: 0.8rem 0.95rem;
            background: rgba(255, 255, 255, 0.025);
        }}

        .warning-box {{
            border: 1px solid rgba(239, 68, 68, 0.28);
            background: rgba(239, 68, 68, 0.08);
            border-radius: 0.9rem;
            padding: 0.8rem 0.95rem;
            color: #fecaca;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_grayscale_numpy(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray.astype(np.float32)


def image_bytes_to_gray(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    return to_grayscale_numpy(image)


def encode_png(image_uint8: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_uint8, mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def bytes_to_kb(data: bytes) -> float:
    return len(data) / 1024.0


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def compute_metrics(original_uint8: np.ndarray, reconstructed_uint8: np.ndarray) -> Dict[str, float]:
    original_f = original_uint8.astype(np.float32)
    reconstructed_f = reconstructed_uint8.astype(np.float32)

    mse = float(np.mean((original_f - reconstructed_f) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    ssim_val = float(ssim(original_uint8, reconstructed_uint8, data_range=255))
    return {"mse": mse, "psnr": psnr, "ssim": ssim_val}


def compression_ratio(original_size_bytes: int, compressed_size_bytes: int) -> float:
    if compressed_size_bytes <= 0:
        return float("inf")
    return float(original_size_bytes / compressed_size_bytes)


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return np.zeros_like(eigenvalues, dtype=np.float32)
    return (eigenvalues / total).astype(np.float32)


def choose_sample_ks(max_k: int, n_points: int = 24) -> List[int]:
    if max_k <= 1:
        return [1]
    n_points = min(n_points, max_k)
    ks = np.unique(np.linspace(1, max_k, num=n_points, dtype=int)).tolist()
    if ks[-1] != max_k:
        ks.append(max_k)
    return sorted(set(int(k) for k in ks))


@st.cache_data(show_spinner=False)
def compute_pca_decomposition(image_bytes: bytes) -> Dict[str, np.ndarray]:
    gray = image_bytes_to_gray(image_bytes).astype(np.float32)

    mean_vector = np.mean(gray, axis=0, dtype=np.float32)
    centered = gray - mean_vector

    covariance = np.cov(centered, rowvar=False).astype(np.float32, copy=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order].astype(np.float32, copy=False)
    eigenvectors = eigenvectors[:, order].astype(np.float32, copy=False)

    scores = centered @ eigenvectors
    ev_ratio = explained_variance_ratio(eigenvalues)
    cumulative = np.cumsum(ev_ratio).astype(np.float32)

    return {
        "gray": gray,
        "mean_vector": mean_vector.astype(np.float32),
        "centered": centered.astype(np.float32),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "scores": scores.astype(np.float32),
        "explained_variance_ratio": ev_ratio,
        "cumulative_explained_variance": cumulative,
    }


def reconstruct_from_pca(pca_result: Dict[str, np.ndarray], k: int) -> np.ndarray:
    eigenvectors = pca_result["eigenvectors"]
    scores = pca_result["scores"]
    mean_vector = pca_result["mean_vector"]

    max_components = eigenvectors.shape[1]
    k = int(np.clip(k, 1, max_components))

    reconstructed = scores[:, :k] @ eigenvectors[:, :k].T + mean_vector
    return normalize_uint8(reconstructed)


@st.cache_data(show_spinner=False)
def build_curve(image_bytes: bytes, ks: Tuple[int, ...]) -> pd.DataFrame:
    pca_result = compute_pca_decomposition(image_bytes)
    original_uint8 = pca_result["gray"].astype(np.uint8)

    rows = []
    for k in ks:
        reconstructed_uint8 = reconstruct_from_pca(pca_result, k)
        comp_bytes = encode_png(reconstructed_uint8)
        metrics = compute_metrics(original_uint8, reconstructed_uint8)
        explained = float(np.sum(pca_result["explained_variance_ratio"][:k]) * 100.0)

        rows.append(
            {
                "k": int(k),
                "explained_variance": explained,
                "mse": metrics["mse"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "compressed_kb": bytes_to_kb(comp_bytes),
                "compressed_bytes": len(comp_bytes),
            }
        )

    return pd.DataFrame(rows)


def plot_scree_plot(eigenvalues: np.ndarray, max_points: int = 40):
    n = min(max_points, len(eigenvalues))
    x = np.arange(1, n + 1)
    y = eigenvalues[:n]

    if not HAS_PLOTLY:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker=dict(color="rgba(34, 211, 238, 0.75)"),
            hovertemplate="PC %{x}<br>Eigenvalue=%{y:.4f}<extra></extra>",
            name="Eigenvalue",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Scree Plot", x=0.02),
        xaxis_title="Principal Component",
        yaxis_title="Eigenvalue",
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
    )
    return fig


def plot_cumulative_variance(cumulative: np.ndarray):
    if not HAS_PLOTLY:
        return None

    x = np.arange(1, len(cumulative) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative,
            mode="lines",
            line=dict(color=CYAN, width=3),
            name="Cumulative Explained Variance",
            hovertemplate="PC %{x}<br>Cumulative=%{y:.4f}<extra></extra>",
        )
    )

    for level, color in [(0.80, AMBER), (0.90, PURPLE), (0.95, GREEN)]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.75)

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Cumulative Explained Variance", x=0.02),
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance",
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def plot_histogram_overlay(original_uint8: np.ndarray, reconstructed_uint8: np.ndarray, title: str):
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=original_uint8.flatten(),
            nbinsx=64,
            opacity=0.62,
            name="Original",
            marker_color=CYAN,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=reconstructed_uint8.flatten(),
            nbinsx=64,
            opacity=0.62,
            name="Reconstructed",
            marker_color=PURPLE,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=title, x=0.02),
        xaxis_title="Pixel Intensity",
        yaxis_title="Count",
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
    )
    return fig


def plot_metric_curves(curve_df: pd.DataFrame):
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=curve_df["k"],
            y=curve_df["psnr"],
            mode="lines+markers",
            line=dict(color=CYAN, width=3),
            marker=dict(size=8),
            name="PSNR",
            hovertemplate="k=%{x}<br>PSNR=%{y:.2f} dB<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df["k"],
            y=curve_df["ssim"],
            mode="lines+markers",
            line=dict(color=PURPLE, width=3),
            marker=dict(size=8),
            name="SSIM",
            yaxis="y2",
            hovertemplate="k=%{x}<br>SSIM=%{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="PSNR and SSIM versus k", x=0.02),
        xaxis_title="k",
        yaxis=dict(title="PSNR (dB)", side="left"),
        yaxis2=dict(
            title="SSIM",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            showgrid=False,
        ),
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


inject_css()

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.2rem;font-weight:800;letter-spacing:-0.02em;color:#f4f7fb;">PCA Image Compression Lab</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="margin-top:0.35rem;color:#dbe7f5;font-size:0.92rem;line-height:1.6;">Upload gambar JPG atau PNG, lalu atur nilai k untuk melihat trade-off kualitas dan ukuran.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1rem 0;'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#eaf2ff;font-size:0.74rem;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;margin-bottom:0.4rem;">K Value</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#b8c4d3;font-size:0.84rem;line-height:1.5;margin-bottom:0.55rem;">Slider ini mengatur jumlah komponen utama yang dipakai saat rekonstruksi.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="color:#eaf2ff;font-size:0.72rem;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;margin-bottom:0.4rem;">Image details</div>',
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.markdown(
        """
        <div style="min-height:72vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
            <div class="hero-title">PCA Image Compression Lab</div>
            <div class="hero-subtitle" style="max-width:720px;">
                Kompresi citra grayscale berbasis Principal Component Analysis dengan evaluasi MSE, PSNR, SSIM, compression ratio, dan visualisasi interaktif.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded_file.getvalue()
pca_result = compute_pca_decomposition(image_bytes)
gray_uint8 = pca_result["gray"].astype(np.uint8)

original_size_bytes = len(image_bytes)
original_size_kb = bytes_to_kb(image_bytes)
max_k = min(pca_result["eigenvectors"].shape[1], pca_result["gray"].shape[1])

if "k_value" not in st.session_state:
    st.session_state["k_value"] = min(8, max_k)

st.session_state["k_value"] = int(np.clip(st.session_state["k_value"], 1, max_k))

with st.sidebar:
    k_value = st.slider(
        "Principal component k",
        min_value=1,
        max_value=max_k,
        value=min(st.session_state["k_value"], max_k),
        step=1,
    )
    st.session_state["k_value"] = k_value

reconstructed_uint8 = reconstruct_from_pca(pca_result, k_value)
compressed_bytes = encode_png(reconstructed_uint8)
compressed_size_bytes = len(compressed_bytes)
compressed_size_kb = bytes_to_kb(compressed_bytes)

selected_metrics = compute_metrics(gray_uint8, reconstructed_uint8)
cr = compression_ratio(original_size_bytes, compressed_size_bytes)
reduction_pct = 100.0 * (1.0 - compressed_size_bytes / original_size_bytes) if original_size_bytes > 0 else 0.0

curve_ks = tuple(choose_sample_ks(max_k=max_k, n_points=24))
curve_df = build_curve(image_bytes, curve_ks)
best_idx = curve_df["ssim"].idxmax()
best_k = int(curve_df.loc[best_idx, "k"])

ev_ratio = pca_result["explained_variance_ratio"]
cumulative = pca_result["cumulative_explained_variance"]
eigenvalues = pca_result["eigenvalues"]

st.markdown(
    f"""
    <div class="panel" style="margin-bottom:1rem;">
        <div class="hero-title">PCA Image Compression Lab</div>
        <div class="hero-subtitle">
            {uploaded_file.name} · {gray_uint8.shape[1]} × {gray_uint8.shape[0]} px · Original size {original_size_kb:.2f} KB
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original size", f"{original_size_kb:.2f} KB")
m2.metric("Compressed size", f"{compressed_size_kb:.2f} KB", delta=f"{reduction_pct:.1f}% smaller", delta_color="inverse")
m3.metric("Compression ratio", f"{cr:.2f}x")
m4.metric("Selected k", f"{k_value}", delta=f"Best k: {best_k}")

tab_overview, tab_eigen, tab_compress, tab_export = st.tabs(
    ["Overview", "Eigen Analysis", "Compression Dashboard", "Export"]
)

with tab_overview:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.markdown('<div class="section-label">Image preview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.image(Image.open(io.BytesIO(image_bytes)), caption="Original image", use_container_width=True)
        with c2:
            st.image(reconstructed_uint8, caption=f"Reconstructed grayscale, k={k_value}", use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Live metrics</div>', unsafe_allow_html=True)
        st.metric("MSE", f"{selected_metrics['mse']:.4f}")
        st.metric("PSNR", f"{selected_metrics['psnr']:.2f} dB")
        st.metric("SSIM", f"{selected_metrics['ssim']:.4f}")
        st.metric("Compression ratio", f"{cr:.2f}x", delta=f"{reduction_pct:.1f}% smaller", delta_color="inverse")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Quick summary</div>', unsafe_allow_html=True)
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Mean intensity", f"{gray_uint8.mean():.2f}")
            st.metric("Std intensity", f"{gray_uint8.std():.2f}")
        with summary_col2:
            st.metric("Output size", f"{compressed_size_kb:.2f} KB")
            st.metric("Explained variance", f"{curve_df.loc[curve_df['k'] == k_value, 'explained_variance'].iloc[0]:.2f}%")

        st.markdown(
            """
            <div class="tech-box">
                <div class="small-note">
                PCA decomposition is cached on the uploaded image bytes. Moving the k slider only triggers reconstruction, not a repeated eigen decomposition.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_eigen:
    st.markdown('<div class="section-label">PCA decomposition</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Top eigenvalue", f"{eigenvalues[0]:.4f}")
    c2.metric("Variance PC1", f"{ev_ratio[0] * 100:.2f}%")
    c3.metric("Variance at selected k", f"{cumulative[min(k_value, len(cumulative)) - 1] * 100:.2f}%")
    c4.metric("Components for 95%", f"{int(np.argmax(cumulative >= 0.95) + 1)}")

    left, right = st.columns(2)
    with left:
        if HAS_PLOTLY:
            st.plotly_chart(plot_scree_plot(eigenvalues), use_container_width=True, config={"displayModeBar": True})
        else:
            st.warning("Plotly belum tersedia. Tambahkan plotly ke requirements untuk grafik interaktif.")
            st.bar_chart(pd.DataFrame({"eigenvalue": eigenvalues[:40]}), use_container_width=True)
    with right:
        if HAS_PLOTLY:
            st.plotly_chart(plot_cumulative_variance(cumulative), use_container_width=True, config={"displayModeBar": True})
        else:
            st.line_chart(pd.DataFrame({"cumulative_variance": cumulative}), use_container_width=True)

    with st.expander("Technical details", expanded=False):
        st.markdown(
            """
            <div class="tech-box">
                <div class="small-note">
                The image is converted to grayscale using NumPy, centered by column mean, then decomposed using eigenvalue and eigenvector analysis on the covariance matrix. Reconstruction uses the first k components.
                </div>
                <div class="small-note" style="margin-top:0.6rem;">
                X̂ = (X - μ)W<sub>k</sub>W<sub>k</sub><sup>T</sup> + μ
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_compress:
    st.markdown('<div class="section-label">Selected reconstruction</div>', unsafe_allow_html=True)

    top_left, top_right = st.columns([1.05, 0.95])

    with top_left:
        st.image(reconstructed_uint8, caption=f"Reconstructed grayscale image for k={k_value}", use_container_width=True)

    with top_right:
        st.markdown('<div class="section-label">Selected k metrics</div>', unsafe_allow_html=True)
        st.metric("MSE", f"{selected_metrics['mse']:.4f}")
        st.metric("PSNR", f"{selected_metrics['psnr']:.2f} dB")
        st.metric("SSIM", f"{selected_metrics['ssim']:.4f}")
        st.metric("Compression ratio", f"{cr:.2f}x", delta=f"{reduction_pct:.1f}% smaller", delta_color="inverse")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Pixel intensity distribution</div>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            st.plotly_chart(
                plot_histogram_overlay(gray_uint8, reconstructed_uint8, "Original versus reconstructed histogram"),
                use_container_width=True,
                config={"displayModeBar": True},
            )
        else:
            st.bar_chart(
                pd.DataFrame(
                    {
                        "original": np.bincount(gray_uint8.flatten(), minlength=256),
                        "reconstructed": np.bincount(reconstructed_uint8.flatten(), minlength=256),
                    }
                ),
                use_container_width=True,
            )

    st.markdown('<div class="section-label">Metrics versus k</div>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(
            plot_metric_curves(curve_df),
            use_container_width=True,
            config={"displayModeBar": True},
        )
    else:
        st.line_chart(curve_df.set_index("k")[["psnr", "ssim"]], use_container_width=True)

    st.markdown('<div class="section-label">Comparison table</div>', unsafe_allow_html=True)
    display_df = curve_df.copy()
    display_df["explained_variance"] = display_df["explained_variance"].round(2)
    display_df["mse"] = display_df["mse"].round(4)
    display_df["psnr"] = display_df["psnr"].round(2)
    display_df["ssim"] = display_df["ssim"].round(4)
    display_df["compressed_kb"] = display_df["compressed_kb"].round(2)

    st.dataframe(
        display_df.rename(
            columns={
                "k": "k",
                "explained_variance": "Explained Variance (%)",
                "mse": "MSE",
                "psnr": "PSNR (dB)",
                "ssim": "SSIM",
                "compressed_kb": "Compressed Size (KB)",
                "compressed_bytes": "Compressed Bytes",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical details", expanded=False):
        st.markdown(
            """
            <div class="tech-box">
                <div class="small-note">
                The curve table is built from the cached PCA result. Only the reconstruction step changes when k changes, which keeps interaction responsive.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_export:
    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown('<div class="section-label">Recommended setting</div>', unsafe_allow_html=True)
        st.metric("Recommended k", f"{best_k}")
        st.metric("Recommended SSIM", f"{float(curve_df.loc[curve_df['k'] == best_k, 'ssim'].iloc[0]):.4f}")
        st.metric("Recommended PSNR", f"{float(curve_df.loc[curve_df['k'] == best_k, 'psnr'].iloc[0]):.2f} dB")
        st.metric("Recommended size", f"{float(curve_df.loc[curve_df['k'] == best_k, 'compressed_kb'].iloc[0]):.2f} KB")

    with right:
        st.markdown('<div class="section-label">Download compressed image</div>', unsafe_allow_html=True)
        st.download_button(
            label=f"Download PNG for k={k_value}",
            data=compressed_bytes,
            file_name=f"pca_compressed_k_{k_value}.png",
            mime="image/png",
        )

        zip_buffer = io.BytesIO()
        with st.spinner("Preparing archive..."):
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for _, row in curve_df.iterrows():
                    k_i = int(row["k"])
                    recon_i = reconstruct_from_pca(pca_result, k_i)
                    zf.writestr(f"pca_k_{k_i}.png", encode_png(recon_i))

                report_lines = [
                    "PCA Image Compression Report",
                    f"File: {uploaded_file.name}",
                    f"Original size: {original_size_kb:.2f} KB",
                    f"Image size: {gray_uint8.shape[1]} x {gray_uint8.shape[0]} px",
                    "",
                    "k | Explained Var (%) | MSE | PSNR (dB) | SSIM | Compressed KB | Compression Ratio",
                    "-" * 86,
                ]
                for _, row in curve_df.iterrows():
                    comp_kb = float(row["compressed_kb"])
                    ratio_i = compression_ratio(original_size_bytes, int(row["compressed_bytes"]))
                    report_lines.append(
                        f"{int(row['k'])} | {float(row['explained_variance']):.2f} | {float(row['mse']):.4f} | "
                        f"{float(row['psnr']):.2f} | {float(row['ssim']):.4f} | {comp_kb:.2f} | {ratio_i:.2f}x"
                    )
                report_lines.append("")
                report_lines.append(f"Recommended k: {best_k}")
                zf.writestr("report.txt", "\n".join(report_lines))

        zip_buffer.seek(0)

        st.download_button(
            label="Download all reconstructions as ZIP",
            data=zip_buffer.getvalue(),
            file_name="pca_compression_results.zip",
            mime="application/zip",
        )

    with st.expander("Technical details", expanded=False):
        st.markdown(
            """
            <div class="tech-box">
                <div class="small-note">
                Compression ratio is computed from the original file bytes divided by the reconstructed image bytes encoded as PNG. This keeps the metric grounded in actual file size, not just array size.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
