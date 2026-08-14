"""utils.py — Fungsi utilitas untuk Dashboard AI Topic Monitor."""
import json
import os
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR    = os.path.join(ROOT, "data", "temp")
LABELS_JSON = os.path.join(ROOT, "models", "topic_labels.json")

# ── Mapping app_source -> nama display yang ramah ─────────────────────────────
_APP_DISPLAY = {
    "chatgpt": "ChatGPT (OpenAI)",
    "grok":    "Grok (xAI)",
    "gemini":  "Google Gemini",
    "bard":    "Google Bard / Gemini",
    "copilot": "Microsoft Copilot",
    "claude":  "Claude (Anthropic)",
    "deepseek":"DeepSeek",
    "perplexity": "Perplexity AI",
    "meta ai": "Meta AI",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_available_apps() -> dict:
    """
    Mendeteksi aplikasi dari data/temp/ (live scraper).
    Setiap file topic_data_{app_id}.csv merepresentasikan satu aplikasi.
    Returns: dict {app_id: display_name}
    """
    apps = {}

    if not os.path.exists(TEMP_DIR):
        return apps

    for filename in os.listdir(TEMP_DIR):
        if not filename.startswith("topic_data_") or not filename.endswith(".csv"):
            continue
        app_id = filename.replace("topic_data_", "").replace(".csv", "")
        try:
            df_check = pd.read_csv(os.path.join(TEMP_DIR, filename), nrows=1)
            if "app_source" in df_check.columns:
                src = str(df_check["app_source"].iloc[0]).strip().lower()
                display = _APP_DISPLAY.get(src, src.title())
            else:
                display = app_id.title()
            apps[app_id] = display
        except Exception:
            apps[app_id] = app_id.title()

    return apps


@st.cache_data(ttl=60)
def load_data(app_id: str) -> pd.DataFrame:
    """
    Memuat data dari data/temp/topic_data_{app_id}.csv.
    Hanya membaca dari folder temp — data final notebook tidak digunakan.
    """
    if not app_id:
        return pd.DataFrame()

    # Hapus prefix __final__ jika ada (backward compat)
    if app_id.startswith("__final__"):
        app_id = app_id.replace("__final__", "")

    path = os.path.join(TEMP_DIR, f"topic_data_{app_id}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["date"])
        except Exception as e:
            st.error(f"Gagal memuat {path}: {e}")

    return pd.DataFrame()


@st.cache_data
def load_topic_labels() -> dict:
    """Memuat mapping ID topik ke label semantik dari topic_labels.json."""
    if not os.path.exists(LABELS_JSON):
        st.warning("⚠️ `models/topic_labels.json` tidak ditemukan. Menggunakan label default.")
        return {"0": "Topik 0", "1": "Topik 1", "2": "Topik 2", "3": "Topik 3"}
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER & GROWTH RATE
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_window(df: pd.DataFrame, window) -> pd.DataFrame:
    """
    Filter DataFrame berdasarkan jendela waktu.
    window: '24H' | '7D' | '30D' | 'Semua Waktu' | tuple(start_date, end_date)
    """
    if df.empty or "date" not in df.columns:
        return df

    if window == "Semua Waktu":
        return df

    if isinstance(window, tuple) and len(window) == 2:
        start_ts = pd.to_datetime(window[0])
        end_ts   = pd.to_datetime(window[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

    now   = df["date"].max()
    delta = {"24H": pd.Timedelta(hours=24), "7D": pd.Timedelta(days=7), "30D": pd.Timedelta(days=30)}
    return df[df["date"] >= now - delta.get(window, pd.Timedelta(days=7))]


def _get_badge(volume: int, max_volume: int) -> str:
    """Menentukan badge berdasarkan volume ulasan tertinggi."""
    if volume == max_volume and volume > 0:
        return "🔥 Hot Topic"
    return "➡️ Stable"


def calculate_growth_rate(df: pd.DataFrame, window) -> pd.DataFrame:
    """
    Menghitung total ulasan per topik dalam jendela waktu yang dipilih.
    Hot Topic ditentukan berdasarkan jumlah ulasan terbanyak.
    Returns DataFrame: topic_label | volume_now | volume_prev | growth_rate | badge
    """
    empty = pd.DataFrame(columns=["topic_label", "volume_now", "volume_prev", "growth_rate", "badge"])

    if df.empty or "topic_label" not in df.columns:
        return empty

    # ── Filter berdasarkan jendela waktu ──────────────────────────────────────
    if window == "Semua Waktu":
        df_now = df
        df_prev = df.iloc[0:0] # Kosong
    elif isinstance(window, tuple) and len(window) == 2:
        start_ts = pd.to_datetime(window[0])
        end_ts   = pd.to_datetime(window[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        duration = end_ts - start_ts + pd.Timedelta(seconds=1)
        df_now   = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        df_prev  = df[(df["date"] >= start_ts - duration) & (df["date"] < start_ts)]
    else:
        now   = df["date"].max()
        delta = {"24H": pd.Timedelta(hours=24), "7D": pd.Timedelta(days=7), "30D": pd.Timedelta(days=30)}
        td    = delta.get(window, pd.Timedelta(days=7))
        df_now   = df[df["date"] >= now - td]
        df_prev  = df[(df["date"] >= now - 2*td) & (df["date"] < now - td)]

    # ── Hitung volume per topik ───────────────────────────────────────────────
    vol_now = df_now["topic_label"].value_counts()
    vol_prev = df_prev["topic_label"].value_counts()

    # Gabungkan volume
    vol = pd.DataFrame({"volume_now": vol_now, "volume_prev": vol_prev}).fillna(0)

    # Hitung laju pertumbuhan (Growth Rate)
    growth_rate = ((vol["volume_now"] - vol["volume_prev"]) / vol["volume_prev"]) * 100
    # Berikan fallback 100% jika pembagi bernilai nol (topik baru muncul)
    import numpy as np
    growth_rate = growth_rate.replace([np.inf, -np.inf], 100.0).fillna(100.0)

    vol["growth_rate"] = growth_rate
    vol.index.name = "topic_label"
    vol = vol.reset_index()

    max_vol = vol["volume_now"].max() if not vol.empty else 0
    vol["badge"] = vol["volume_now"].apply(lambda v: _get_badge(v, max_vol))

    return vol.sort_values(by="volume_now", ascending=False).reset_index(drop=True)
