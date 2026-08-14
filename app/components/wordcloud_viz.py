"""wordcloud_viz.py — Komponen visualisasi Word Cloud untuk Dashboard Trend24."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import numpy as np


# ── Palet: dark background word cloud ─────────────────────────────────────────
_COLORMAP = "plasma"
_BG_COLOR = "#111827"


def _circular_mask(size: int = 400) -> np.ndarray:
    """Membuat mask lingkaran untuk bentuk word cloud yang rapi."""
    x, y = np.ogrid[:size, :size]
    mask = (x - size // 2) ** 2 + (y - size // 2) ** 2 > (size // 2 - 5) ** 2
    # WordCloud menggunakan 255 untuk area yang di-mask (dilewati)
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[mask] = 255
    return arr


def generate_wordcloud(df: pd.DataFrame, topic_label = None) -> None:
    """
    Menampilkan word cloud untuk satu topik tertentu atau seluruh topik.

    Args:
        df: DataFrame yang sudah difilter berdasarkan jendela waktu.
        topic_label: Label topik (str) atau None untuk menampilkan semua ulasan.
    """
    # ── 1. Filter data sesuai topik ──────────────────────────────────────────
    if topic_label is not None:
        subset = df[df['topic_label'] == topic_label]
    else:
        subset = df  # Semua topik

    if subset.empty:
        st.warning(f"⚠️ Tidak ada data untuk topik: **{topic_label or 'Semua Topik'}**")
        return

    # Cek kolom teks — gunakan text_clean jika ada, fallback ke text
    text_col = 'text_clean' if 'text_clean' in subset.columns else 'text'
    text_data = " ".join(subset[text_col].astype(str).dropna().tolist()).strip()

    if not text_data:
        st.warning(f"⚠️ Teks kosong untuk topik: **{topic_label}**")
        return

    # ── 2. Generate Word Cloud ───────────────────────────────────────────────
    mask = _circular_mask(500)

    wc = WordCloud(
        width=800,
        height=450,
        background_color=_BG_COLOR,
        colormap=_COLORMAP,
        max_words=120,
        mask=mask,
        contour_width=1,
        contour_color="#6366f1",
        prefer_horizontal=0.85,
        random_state=42,
        min_font_size=10,
        max_font_size=80,
        relative_scaling=0.6
    ).generate(text_data)

    # ── 3. Plot dengan Matplotlib ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=_BG_COLOR)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_facecolor(_BG_COLOR)

    # Metadata di bawah gambar
    n_docs = len(subset)
    n_words = len(text_data.split())
    top_words = list(wc.words_.keys())[:5]
    top_str = " · ".join(top_words) if top_words else "—"

    fig.text(
        0.5, 0.02,
        f"{n_docs:,} ulasan  |  {n_words:,} token  |  Top: {top_str}",
        ha='center', va='bottom',
        fontsize=8, color='#64748b',
        fontfamily='monospace'
    )

    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)

    # ── 4. Top-10 kata kunci (frekuensi aktual dari teks) ─────────────────
    with st.expander("📋 Lihat Top-10 Kata Kunci", expanded=False):
        # Hitung frekuensi aktual kata dari teks
        words = text_data.split()
        word_counts = pd.Series(words).value_counts().head(10)
        if not word_counts.empty:
            df_top = word_counts.reset_index()
            df_top.columns = ["Kata", "Jumlah"]
            df_top.index += 1
            st.dataframe(df_top, use_container_width=True, hide_index=False)
