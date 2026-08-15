"""main.py — Entry point Dashboard Monitoring (Streamlit) — Versi UI/UX Premium."""
import os
import sys
import streamlit as st
import pandas as pd

# Pastikan direktori app/ ada di sys.path (aman untuk streamlit_app.py dan app/main.py)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ── Konfigurasi Halaman (HARUS di baris pertama) ──────────────────────────────
st.set_page_config(
    page_title="AI App Topic Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── CSS Kustom ────────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Import Modul Lokal ────────────────────────────────────────────────────────
from utils import (
    load_topic_labels, filter_by_window, calculate_growth_rate,
)
from components.charts import (
    plot_trend_line, plot_topic_distribution,
    plot_rating_distribution, plot_topic_heatmap,
)
from components.wordcloud_viz import generate_wordcloud
from live_scraper import fetch_and_process_reviews, search_app_in_playstore

# ── Session State ─────────────────────────────────────────────────────────────
if "selected_topic_wc" not in st.session_state:
    st.session_state.selected_topic_wc = None
if "selected_app_id" not in st.session_state:
    st.session_state.selected_app_id = None
if "_just_reset" not in st.session_state:
    st.session_state._just_reset = False
# Dataset per-session: data hanya hidup di session ini, hilang saat refresh.
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if "df_data_app_id" not in st.session_state:
    st.session_state.df_data_app_id = None
if "df_data_app_name" not in st.session_state:
    st.session_state.df_data_app_name = None

# ── Helper: simpan dataset ke session (bukan file) ───────────────────────────
def save_df_to_session(df_new: pd.DataFrame, app_id: str, app_name: str) -> int:
    """Simpan DataFrame ke session state. Overwrite total (reset on new fetch)."""
    base_cols = ["date", "text", "text_clean", "rating", "topic_label", "topic_probability",
                 "app_source", "dominant_topic"]
    extra_cols = [c for c in df_new.columns if c.startswith("prob_")]
    all_cols   = base_cols + extra_cols
    df_clean   = df_new[[c for c in all_cols if c in df_new.columns]].copy()
    st.session_state.df_data       = df_clean
    st.session_state.df_data_app_id   = app_id
    st.session_state.df_data_app_name = app_name
    return len(df_clean)

# ── Load Label Topik ──────────────────────────────────────────────────────────
try:
    topic_labels = load_topic_labels()
    all_topics   = sorted(list(topic_labels.values()))
except Exception as e:
    st.error(f"⚠️ **Gagal memuat label topik:** `{e}`")
    all_topics = []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: Live Scraper
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📡 Live Data Fetching")
    st.markdown("Cari aplikasi di Play Store dan ambil ulasan terbarunya.")

    search_query = st.text_input("🔍 Cari Aplikasi:", placeholder="Contoh: Grok, Gemini, DeepSeek...")
    if search_query:
        apps_found = search_app_in_playstore(search_query)
        if apps_found:
            app_options      = {app["appId"]: app.get("title", app["appId"]) for app in apps_found if app.get("appId")}
            scraper_app_id   = st.selectbox(
                "Hasil Pencarian:", options=list(app_options.keys()),
                format_func=lambda x: app_options[x],
            )
            limit_choice = st.number_input(
                "Jumlah Ulasan:",
                min_value=10, max_value=10000, value=200, step=50,
                help="Peringatan: Mengambil >1000 ulasan mungkin memakan waktu beberapa menit."
            )

            if st.button("🔄 Fetch & Update Dataset", use_container_width=True):
                with st.spinner("Sedang memproses... (Normalisasi & Inferensi Topik)"):
                    df_new = fetch_and_process_reviews(scraper_app_id, app_options[scraper_app_id], limit_choice)
                    if not df_new.empty:
                        added = save_df_to_session(df_new, scraper_app_id, app_options[scraper_app_id])
                        st.session_state.selected_app_id = scraper_app_id
                        st.cache_data.clear()
                        notif_placeholder = st.empty()
                        if added > 0:
                            notif_placeholder.success(f"✅ {added} ulasan dimuat ke dataset sesi ini.")
                        else:
                            notif_placeholder.warning("Data kosong.")
                        import time
                        time.sleep(5)
                        notif_placeholder.empty()
                        st.rerun()
        else:
            st.info("Pencarian tidak menemukan hasil.")

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
            <h1 class="hero-title">🤖 AI App Topic Monitor</h1>
            <p class="hero-subtitle">
                Sistem Monitoring Tren Topik Ulasan Pengguna Aplikasi AI
            </p>
            <span class="hero-badge">● LIVE</span>
        </div>
        <div style="text-align:right; color:rgba(255,255,255,0.4); font-size:0.8rem; line-height:1.8;">
            <div>📊 Model: Topic Modelling LDA</div>
            <div>🌐 Sumber: Google Play Store (ID)</div>
            <div>📚 Bahasa: Indonesia</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER BAR — dataset dari session state (per-session, bukan file)
# ─────────────────────────────────────────────────────────────────────────────
df_session = st.session_state.df_data
session_app_id   = st.session_state.df_data_app_id
session_app_name = st.session_state.df_data_app_name

# Auto-load session app hanya saat awal, bukan setelah reset
if session_app_id and st.session_state.selected_app_id is None and not st.session_state._just_reset:
    st.session_state.selected_app_id = session_app_id
st.session_state._just_reset = False  # clear flag setelah dipakai

selected_app_id = st.session_state.selected_app_id

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc0, fc1, fc2 = st.columns([2.0, 2.5, 1.5])

    with fc0:
        if selected_app_id and not df_session.empty:
            app_display_name = session_app_name or selected_app_id
            st.markdown(f"📱 **Nama Aplikasi**<br><span style='font-size:1.15rem; font-weight:bold; color:#fdfdfd;'>{app_display_name}</span>", unsafe_allow_html=True)
            df_raw = df_session
        else:
            st.markdown("📱 **Nama Aplikasi**<br><span style='color:rgba(255,255,255,0.4);'>Tidak ada data</span>", unsafe_allow_html=True)
            df_raw = pd.DataFrame()

    with fc1:
        selected_topics = st.multiselect(
            "🎯 **Filter Topik**",
            options=all_topics, default=all_topics,
            help="Pilih satu atau beberapa topik.",
        )

    with fc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Dataset", use_container_width=True):
            st.session_state.df_data       = pd.DataFrame()
            st.session_state.df_data_app_id   = None
            st.session_state.df_data_app_name = None
            st.session_state.selected_app_id = None
            st.session_state._just_reset = True
            st.cache_data.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ── Guard: hentikan jika data kosong ─────────────────────────────────────────
if df_raw.empty:
    st.info("👋 Mulailah dengan fetch data baru dari Sidebar.")
    st.stop()

# ── Filter & Hitung Growth ────────────────────────────────────────────────────
growth_df   = calculate_growth_rate(df_raw, "Semua Waktu")
df_filtered = df_raw
df_display  = (
    df_filtered[df_filtered["topic_label"].isin(selected_topics)]
    if selected_topics else df_filtered
)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC TILES
# ─────────────────────────────────────────────────────────────────────────────
total_reviews = len(df_display)
active_topics = df_display["topic_label"].nunique()

hot_row   = growth_df[growth_df["volume_now"] > 0].iloc[0] if not growth_df.empty else None
hot_name  = hot_row["topic_label"] if hot_row is not None else "N/A"
hot_vol   = int(hot_row["volume_now"]) if hot_row is not None else 0

def _fmt_range(start, end) -> str:
    """Format rentang tanggal secara ringkas: tahun hanya muncul jika berbeda tahun."""
    if hasattr(start, "year") and hasattr(end, "year"):
        if start.year == end.year:
            return f"{start.strftime('%d %b')} – {end.strftime('%d %b %Y')}"
        else:
            return f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}"
    return f"{start} – {end}"

if not df_raw.empty and "date" in df_raw.columns:
    _min = pd.to_datetime(df_raw["date"].min())
    _max = pd.to_datetime(df_raw["date"].max())
    window_label = _fmt_range(_min, _max)
else:
    window_label = "Semua Waktu"

def _metric_html(col, icon_label: str, value: str):
    col.markdown(f"""
<div style="
    background: linear-gradient(145deg, #1a1f2e, #1e2538);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    height: 100%;
">
    <div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;
                letter-spacing:0.06em;color:rgba(255,255,255,0.45);margin-bottom:0.5rem;">
        {icon_label}
    </div>
    <div style="font-size:1.7rem;font-weight:700;color:#ffffff;line-height:1.2;word-break:break-word;">
        {value}
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
_metric_html(m1, "🏷️ Topik Aktif",   str(active_topics))
_metric_html(m2, "⏱️ Rentang Tanggal", window_label)
_metric_html(m3, "🗄️ Total Ulasan", f"{len(df_raw):,}")

# Hot Topic Banner
st.markdown(f"""
<div style="margin-top:1rem; padding:0.9rem 1.4rem;
    background:linear-gradient(135deg,rgba(255,100,30,0.15),rgba(255,60,10,0.05));
    border:1px solid rgba(255,100,30,0.4); border-left:4px solid #ff6920;
    border-radius:10px; display:flex; align-items:center; gap:1rem;">
    <span style="font-size:1.6rem;">🔥</span>
    <div>
        <div style="font-size:0.75rem;color:#a0aec0;margin-bottom:2px;letter-spacing:.05em;text-transform:uppercase;">Hot Topic (Ulasan Terbanyak)</div>
        <div style="font-size:1.1rem;font-weight:700;color:#fdfdfd;">{hot_name} — <span style="color:#ff6920;">{hot_vol:,} ulasan</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ROW A: Trend Line + Rating Distribution
# ─────────────────────────────────────────────────────────────────────────────
col_trend, col_rating = st.columns([2, 1], gap="large")

with col_trend:
    st.markdown('<p class="section-title">📈 Tren Volume Topik</p>', unsafe_allow_html=True)
    fig_trend = plot_trend_line(df_display, "Semua Waktu")
    if fig_trend:
        st.plotly_chart(fig_trend, config={"displayModeBar": False}, use_container_width=True)
    else:
        st.info("📭 Tidak ada data tren untuk filter ini.")

with col_rating:
    st.markdown('<p class="section-title">⭐ Distribusi Rating</p>', unsafe_allow_html=True)
    fig_rating = plot_rating_distribution(df_display)
    if fig_rating:
        st.plotly_chart(fig_rating, config={"displayModeBar": False}, use_container_width=True)
    else:
        st.info("Tidak ada data rating.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ROW B: Topic Cards + Word Cloud
# ─────────────────────────────────────────────────────────────────────────────
col_cards, col_wc = st.columns([1, 1], gap="large")

with col_cards:
    st.markdown('<p class="section-title">🏷️ Kartu Topik &amp; Total Ulasan</p>', unsafe_allow_html=True)
    filtered_growth = (
        growth_df[growth_df["topic_label"].isin(selected_topics)]
        if selected_topics else growth_df
    )

    if filtered_growth.empty:
        st.info("Tidak ada data untuk topik yang dipilih.")
    else:
        # ── Tombol "Semua Topik" ──────────────────────────────────────────
        is_all_active = st.session_state.selected_topic_wc is None
        if st.button(f"☁️ Semua Topik", key="wc_all", use_container_width=True):
            st.session_state.selected_topic_wc = None
            st.rerun()

        for _, row in filtered_growth.iterrows():
            vol        = int(row["volume_now"])
            is_active  = st.session_state.selected_topic_wc == row["topic_label"]

            st.markdown(f"""
            <div class="topic-card stable{'  active' if is_active else ''}">
                <div class="topic-card-name">{row['topic_label']}</div>
                <div class="topic-card-volume">📄 {vol:,} ulasan</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"☁️ Lihat Word Cloud", key=f"wc_{row['topic_label']}", use_container_width=True):
                st.session_state.selected_topic_wc = row["topic_label"]
                st.rerun()

with col_wc:
    st.markdown('<p class="section-title">☁️ Word Cloud Topik</p>', unsafe_allow_html=True)

    wc_topic = st.session_state.selected_topic_wc  # None = Semua Topik

    if wc_topic is None:
        st.caption("Menampilkan kata kunci dominan untuk: **Semua Topik**")
        generate_wordcloud(df_display)  # None → semua ulasan
        st.markdown("---")
        st.markdown("**Distribusi Topik (Periode Ini)**")
        pie_fig = plot_topic_distribution(df_display)
        if pie_fig:
            st.plotly_chart(pie_fig, config={"displayModeBar": False}, use_container_width=True)
    elif wc_topic in (selected_topics or []):
        st.caption(f"Menampilkan kata kunci dominan untuk: **{wc_topic}**")
        generate_wordcloud(df_display, wc_topic)
        st.markdown("---")
        st.markdown("**Distribusi Topik (Periode Ini)**")
        pie_fig = plot_topic_distribution(df_display)
        if pie_fig:
            st.plotly_chart(pie_fig, config={"displayModeBar": False}, use_container_width=True)
    else:
        st.info("💡 Pilih 'Semua Topik' atau topik spesifik untuk menampilkan Word Cloud.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ROW C: Heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig_heatmap = plot_topic_heatmap(df_display)
if fig_heatmap:
    st.markdown('<p class="section-title">🗓️ Heatmap Volume per Bulan</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_heatmap, config={"displayModeBar": False}, use_container_width=True)
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RAW EVIDENCE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📋 Raw Evidence — Ulasan Asli</p>', unsafe_allow_html=True)

ev_col1, ev_col2, ev_col3 = st.columns([2, 1.2, 1.8])
with ev_col1:
    search = st.text_input(
        "🔍 Cari kata kunci...",
        placeholder="Contoh: lambat, crash, login...",
        label_visibility="collapsed",
    )
with ev_col2:
    topic_filter_ev = st.selectbox(
        "Filter Topik",
        options=["Semua Topik"] + (selected_topics if selected_topics else all_topics),
        label_visibility="collapsed",
    )
with ev_col3:
    if not df_display.empty and "date" in df_display.columns:
        _min_d = df_display["date"].min().date()
        _max_d = df_display["date"].max().date()
        ev_date_range = st.date_input(
            "Filter Tanggal",
            value=(_min_d, _max_d),
            min_value=_min_d,
            max_value=_max_d,
            label_visibility="collapsed"
        )
    else:
        ev_date_range = None

evidence_cols  = ["date", "text", "text_clean", "rating", "topic_label", "topic_probability"]
available_cols = [c for c in evidence_cols if c in df_display.columns]
df_ev = df_display[available_cols].sort_values(by="date", ascending=False)

if topic_filter_ev != "Semua Topik":
    df_ev = df_ev[df_ev["topic_label"] == topic_filter_ev]
if search:
    df_ev = df_ev[df_ev["text"].str.contains(search, case=False, na=False)]
if ev_date_range and isinstance(ev_date_range, (tuple, list)) and len(ev_date_range) == 2:
    df_ev = df_ev[(df_ev["date"].dt.date >= ev_date_range[0]) & (df_ev["date"].dt.date <= ev_date_range[1])]

# Pagination
items_per_page = 100
total_items = len(df_ev)
total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

if "ev_page" not in st.session_state:
    st.session_state.ev_page = 1

if st.session_state.ev_page > total_pages:
    st.session_state.ev_page = total_pages
if st.session_state.ev_page < 1:
    st.session_state.ev_page = 1

start_idx = (st.session_state.ev_page - 1) * items_per_page
end_idx = min(start_idx + items_per_page, total_items)
df_ev_page = df_ev.iloc[start_idx:end_idx]

col_config = {}
if "date"              in available_cols: col_config["date"]              = st.column_config.DatetimeColumn("📅 Tanggal", format="DD MMM YYYY, HH:mm", width="medium")
if "text"              in available_cols: col_config["text"]              = st.column_config.TextColumn("💬 Ulasan Asli", width="medium")
if "text_clean"        in available_cols: col_config["text_clean"]        = st.column_config.TextColumn("✨ Hasil Pembersihan", width="medium")
if "rating"            in available_cols: col_config["rating"]            = st.column_config.NumberColumn("⭐ Rating", width="small", format="%d ⭐")
if "topic_label"       in available_cols: col_config["topic_label"]       = st.column_config.TextColumn("🏷️ Topik", width="medium")
if "topic_probability" in available_cols: col_config["topic_probability"] = st.column_config.ProgressColumn("📊 Probabilitas", min_value=0, max_value=1, width="small")

st.dataframe(df_ev_page, column_config=col_config, hide_index=True, height=380, use_container_width=True)

# Page controls layout (page numbers only, left-aligned, and small size)
pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1.1, 0.9, 4.0, 6.0])
with pag_col1:
    st.markdown("<p style='margin-top:0.45rem; font-size:0.85rem; text-align:left; font-weight:600; color:rgba(255,255,255,0.7);'>Pilih Halaman:</p>", unsafe_allow_html=True)
with pag_col2:
    selected_page = st.selectbox(
        "Halaman",
        options=list(range(1, total_pages + 1)),
        index=min(st.session_state.ev_page - 1, total_pages - 1),
        label_visibility="collapsed"
    )
    if selected_page != st.session_state.ev_page:
        st.session_state.ev_page = selected_page
        st.rerun()
with pag_col3:
    st.markdown(
        f"<p style='margin-top:0.4rem; font-size:0.8rem; color:rgba(255,255,255,0.5);'>"
        f"Halaman <strong>{selected_page}</strong> dari <strong>{total_pages}</strong> (Menampilkan {start_idx+1}–{end_idx} dari {total_items} ulasan)"
        f"</p>",
        unsafe_allow_html=True
    )
with pag_col4:
    csv_data = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset",
        data=csv_data,
        file_name=f"topic_data_{selected_app_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
pass
