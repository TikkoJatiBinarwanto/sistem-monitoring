"""charts.py — Komponen visualisasi grafik untuk Dashboard AI Topic Monitor."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Palet Warna Konsisten per Topik ──────────────────────────────────────────
TOPIC_PALETTE = [
    "#6366f1",  # Indigo  — Topik 0
    "#10b981",  # Emerald — Topik 1
    "#f59e0b",  # Amber   — Topik 2
    "#ef4444",  # Red     — Topik 3
    "#8b5cf6",  # Violet  — Topik 4+
    "#06b6d4",  # Cyan
]

_DARK_BG    = "#0f172a"
_GRID_COLOR = "rgba(255,255,255,0.05)"
_FONT_COLOR = "#94a3b8"
_BASE_LAYOUT = dict(
    plot_bgcolor=_DARK_BG,
    paper_bgcolor=_DARK_BG,
    font=dict(family="Inter, sans-serif", color=_FONT_COLOR, size=12),
)


def _color_map(topics: list) -> dict:
    return {t: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, t in enumerate(sorted(topics))}


# ─────────────────────────────────────────────────────────────────────────────
def plot_trend_line(df: pd.DataFrame, window: str):
    """
    Multi-line area chart volume topik per periode waktu.
    Returns: Plotly Figure | None
    """
    if df.empty or "date" not in df.columns or "topic_label" not in df.columns:
        return None

    freq_map = {"24H": "h", "7D": "D", "30D": "D"}
    freq     = freq_map.get(window, "D") if isinstance(window, str) else "D"

    window_labels = {
        "24H": "24 Jam Terakhir", "7D": "7 Hari Terakhir",
        "30D": "30 Hari Terakhir", "Semua Waktu": "Semua Waktu",
    }
    window_str = window if isinstance(window, str) else "Rentang Tanggal"

    df_plot = df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])

    df_trend = (
        df_plot.set_index("date")
        .groupby("topic_label")
        .resample(freq)["text"]
        .count()
        .reset_index()
    )
    df_trend.columns = ["topic_label", "date", "volume"]

    if df_trend.empty:
        return None

    topics    = sorted(df_trend["topic_label"].unique().tolist())
    cmap      = _color_map(topics)
    fig       = go.Figure()

    for topic in topics:
        subset = df_trend[df_trend["topic_label"] == topic]
        color  = cmap[topic]
        # Parse hex to RGB for fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.08)"

        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["volume"],
            mode="lines+markers",
            name=topic,
            line=dict(width=2.5, color=color, shape="spline"),
            marker=dict(size=5, color=color),
            hovertemplate=(
                f"<b>{topic}</b><br>"
                "Waktu: %{x|%d %b %Y}<br>"
                "Volume: %{y} ulasan<extra></extra>"
            ),
            fill="tozeroy",
            fillcolor=fill_color,
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(gridcolor=_GRID_COLOR, title=None, tickfont=dict(size=11), showline=False),
        yaxis=dict(gridcolor=_GRID_COLOR, title="Jumlah Ulasan", tickfont=dict(size=11),
                   showline=False, zeroline=False),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        height=360,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_topic_distribution(df: pd.DataFrame):
    """
    Donut chart distribusi topik dominan dalam korpus yang ditampilkan.
    Returns: Plotly Figure | None
    """
    if df.empty or "topic_label" not in df.columns:
        return None

    dist = df["topic_label"].value_counts().reset_index()
    dist.columns = ["topic_label", "count"]
    if dist.empty:
        return None

    topics = sorted(dist["topic_label"].tolist())
    colors = [_color_map(topics)[t] for t in dist["topic_label"]]
    total  = dist["count"].sum()

    fig = go.Figure(go.Pie(
        labels=dist["topic_label"],
        values=dist["count"],
        hole=0.60,
        marker=dict(colors=colors, line=dict(color=_DARK_BG, width=2)),
        textinfo="percent",
        textfont=dict(size=11, family="Inter"),
        hovertemplate="<b>%{label}</b><br>%{value:,} ulasan (%{percent})<extra></extra>",
        sort=False,
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )

    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px'>ulasan</span>",
        x=0.5, y=0.5,
        font=dict(size=16, color="#e2e8f0", family="Inter"),
        showarrow=False, align="center",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_rating_distribution(df: pd.DataFrame):
    """
    Bar chart distribusi rating (1-5 bintang) dari data yang ditampilkan.
    Returns: Plotly Figure | None
    """
    if df.empty or "rating" not in df.columns:
        return None

    dist = df["rating"].value_counts().sort_index().reset_index()
    dist.columns = ["rating", "count"]

    star_colors = {1: "#ef4444", 2: "#f97316", 3: "#f59e0b", 4: "#84cc16", 5: "#10b981"}
    colors = [star_colors.get(int(r), "#6366f1") for r in dist["rating"]]

    fig = go.Figure(go.Bar(
        x=dist["rating"].astype(str) + " ⭐",
        y=dist["count"],
        marker_color=colors,
        text=dist["count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="Rating %{x}<br>%{y:,} ulasan<extra></extra>",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(gridcolor=_GRID_COLOR, title=None),
        yaxis=dict(gridcolor=_GRID_COLOR, title="Jumlah Ulasan", showline=False, zeroline=False),
        height=280,
        bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_topic_heatmap(df: pd.DataFrame):
    """
    Heatmap: topik (baris) × bulan (kolom), nilai = volume ulasan.
    Returns: Plotly Figure | None
    """
    if df.empty or "date" not in df.columns or "topic_label" not in df.columns:
        return None

    df_h = df.copy()
    df_h["date"] = pd.to_datetime(df_h["date"])
    df_h["month"] = df_h["date"].dt.to_period("M").astype(str)

    pivot = (
        df_h.groupby(["topic_label", "month"])
        .size()
        .unstack(fill_value=0)
    )

    if pivot.empty or pivot.shape[1] < 2:
        return None

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Purp",
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:,} ulasan<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=_FONT_COLOR), len=0.8),
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title=None, tickfont=dict(size=10), tickangle=-30),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        height=300,
        margin=dict(l=0, r=40, t=50, b=60),
    )
    return fig
