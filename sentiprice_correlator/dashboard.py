"""
dashboard.py — Production-Ready Streamlit Dashboard

Features
────────
  • Dark-themed, glassmorphism UI with premium aesthetics
  • Real-time backend health monitoring
  • 4 animated KPI metric cards (price, prediction, change, sentiment)
  • Dual-axis chart: Price + Sentiment overlay
  • Volume bar chart
  • Model info panel
  • Downloadable raw data
  • Full API integration (/health, /predict, /history, /model/info)

Run
───
  streamlit run dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ===================================================================== #
#  Page Config
# ===================================================================== #
st.set_page_config(
    page_title="Sentiprice Correlator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================== #
#  API Config
# ===================================================================== #
API_BASE = "http://127.0.0.1:8000"

# ===================================================================== #
#  Custom CSS — Dark Theme + Glassmorphism
# ===================================================================== #
st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Header Gradient Bar ────────────────────────────────────── */
    .header-bar {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .header-bar h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-bar p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* ── Glass Card ─────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    /* ── Metric Card ────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(79, 70, 229, 0.15);
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .delta-positive { color: #34d399; }
    .delta-negative { color: #f87171; }
    .delta-neutral { color: #fbbf24; }
    .color-white { color: #ffffff; }
    .color-green { color: #34d399; }
    .color-red { color: #f87171; }
    .color-blue { color: #60a5fa; }
    .color-amber { color: #fbbf24; }

    /* ── Status Dot ─────────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }
    .status-online { background: #34d399; box-shadow: 0 0 8px #34d399; }
    .status-offline { background: #f87171; box-shadow: 0 0 8px #f87171; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ── Section Title ──────────────────────────────────────────── */
    .section-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Hide default Streamlit elements ────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Plotly chart container ──────────────────────────────────── */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================== #
#  Helper Functions
# ===================================================================== #
def api_get(endpoint: str, params: dict = None):
    """Safe GET request to the backend."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to API. Is main.py running?"
    except Exception as e:
        return None, str(e)


def api_post(endpoint: str, json_data: dict):
    """Safe POST request to the backend."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to API. Is main.py running?"
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def sentiment_color(score: float) -> str:
    if score >= 0.3:
        return "color-green"
    elif score <= -0.3:
        return "color-red"
    return "color-amber"


def sentiment_emoji(score: float) -> str:
    if score >= 0.3:
        return "🟢 Bullish"
    elif score <= -0.3:
        return "🔴 Bearish"
    return "🟡 Neutral"


def delta_class(value: float) -> str:
    if value > 0:
        return "delta-positive"
    elif value < 0:
        return "delta-negative"
    return "delta-neutral"


# ===================================================================== #
#  Sidebar
# ===================================================================== #
with st.sidebar:
    st.markdown("### 🧠 Sentiprice")
    st.markdown("---")

    # ── Backend Health ───────────────────────────────────────────
    health, health_err = api_get("/health")
    if health:
        dot_cls = "status-online" if health["status"] == "healthy" else "status-offline"
        st.markdown(
            f'<span class="status-dot {dot_cls}"></span> '
            f'**API:** {health["status"].capitalize()} &nbsp;·&nbsp; `{health["device"]}`',
            unsafe_allow_html=True,
        )
        st.caption(f"Uptime: {health['uptime_seconds']:.0f}s")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span> **API:** Offline',
            unsafe_allow_html=True,
        )
        st.caption("Start the API with: `python main.py`")

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────
    ticker = st.text_input("📈 Ticker Symbol", value="AAPL", key="ticker_input")
    days = st.slider("📅 History (days)", min_value=2, max_value=60, value=14, key="days_slider")
    analyze_btn = st.button("⚡ Analyze", use_container_width=True, type="primary")

    st.markdown("---")

    # ── Model Info ───────────────────────────────────────────────
    with st.expander("🔧 Model Info"):
        model_info, info_err = api_get("/model/info")
        if model_info:
            st.code(f"Architecture: {model_info['architecture']}")
            st.code(f"Parameters: {model_info['total_parameters']:,}")
            st.code(f"Features: {len(model_info['feature_names'])}")
            if model_info.get("training_config"):
                cfg = model_info["training_config"]
                st.caption(
                    f"Trained on: {cfg.get('ticker', 'N/A')} · "
                    f"LR: {cfg.get('lr', 'N/A')} · "
                    f"Epochs: {cfg.get('epochs', 'N/A')}"
                )
        else:
            st.caption("Model info unavailable")

    st.markdown("---")
    st.caption("v1.0.0 · Built with Streamlit + FastAPI")


# ===================================================================== #
#  Header
# ===================================================================== #
st.markdown("""
<div class="header-bar">
    <h1>🤖 Multi-Modal Sentiment & Price Correlator</h1>
    <p>BERT-powered news sentiment analysis fused with LSTM time-series prediction</p>
</div>
""", unsafe_allow_html=True)


# ===================================================================== #
#  Main Content
# ===================================================================== #
if analyze_btn:
    # ── Prediction ───────────────────────────────────────────────
    with st.spinner(f"Analyzing {ticker}..."):
        pred_data, pred_err = api_post("/predict", {"ticker": ticker, "days": days})

    if pred_err:
        st.error(f"⚠️ Prediction Error: {pred_err}")
    elif pred_data:
        # ── KPI Metric Cards ─────────────────────────────────────
        st.markdown('<div class="section-title">📊 Prediction Summary</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value color-white">${pred_data['current_price']:,.2f}</div>
                <div class="metric-delta delta-neutral">{ticker}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            chg = pred_data["price_change"]
            chg_cls = delta_class(chg)
            arrow = "▲" if chg > 0 else "▼" if chg < 0 else "●"
            val_color = "color-green" if chg > 0 else "color-red" if chg < 0 else "color-white"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted (1hr)</div>
                <div class="metric-value {val_color}">${pred_data['predicted_next_hour']:,.2f}</div>
                <div class="metric-delta {chg_cls}">{arrow} ${abs(chg):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            pct = pred_data["price_change_pct"]
            pct_cls = delta_class(pct)
            pct_arrow = "▲" if pct > 0 else "▼" if pct < 0 else "●"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price Change</div>
                <div class="metric-value {pct_cls}">{pct_arrow} {abs(pct):.4f}%</div>
                <div class="metric-delta delta-neutral">Next Hour</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            sent = pred_data["sentiment_score"]
            sent_cls = sentiment_color(sent)
            sent_label = sentiment_emoji(sent)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value {sent_cls}">{sent:.4f}</div>
                <div class="metric-delta delta-neutral">{sent_label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical Charts ────────────────────────────────────────
    with st.spinner("Loading historical data..."):
        hist_data, hist_err = api_get(f"/history/{ticker}", params={"days": days})

    if hist_err:
        st.warning(f"⚠️ History Error: {hist_err}")
    elif hist_data and hist_data.get("points"):
        df = pd.DataFrame(hist_data["points"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # ── Price + Sentiment Dual-Axis Chart ────────────────────
        st.markdown('<div class="section-title">📈 Price vs Sentiment Timeline</div>', unsafe_allow_html=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Price area fill
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                name="Close Price",
                mode="lines",
                line=dict(color="#818cf8", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(129, 140, 248, 0.08)",
                hovertemplate="$%{y:,.2f}<extra>Close</extra>",
            ),
            secondary_y=False,
        )

        # Sentiment markers (color-coded)
        colors = [
            "#34d399" if s >= 0.3 else "#f87171" if s <= -0.3 else "#fbbf24"
            for s in df["sentiment"]
        ]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sentiment"],
                name="Sentiment",
                mode="lines+markers",
                line=dict(color="rgba(251, 191, 36, 0.4)", width=1.5),
                marker=dict(color=colors, size=5, line=dict(width=0)),
                hovertemplate="%{y:.3f}<extra>Sentiment</extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            template="plotly_dark",
            height=480,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
            ),
            hovermode="x unified",
        )
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
        )
        fig.update_yaxes(
            title_text="Price ($)",
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Sentiment",
            range=[-1.1, 1.1],
            gridcolor="rgba(255,255,255,0.02)",
            showgrid=False,
            secondary_y=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Volume Bar Chart ─────────────────────────────────────
        st.markdown('<div class="section-title">📊 Volume</div>', unsafe_allow_html=True)

        vol_colors = [
            "#34d399" if c >= p else "#f87171"
            for c, p in zip(df["close"].iloc[1:], df["close"].iloc[:-1])
        ]
        vol_colors.insert(0, "#60a5fa")  # first bar default

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                marker_color=vol_colors,
                marker_line_width=0,
                opacity=0.7,
                hovertemplate="%{y:,.0f}<extra>Volume</extra>",
            )
        )
        fig_vol.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
            showlegend=False,
            hovermode="x unified",
        )
        fig_vol.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        fig_vol.update_yaxes(gridcolor="rgba(255,255,255,0.04)")

        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Statistics Summary ───────────────────────────────────
        st.markdown('<div class="section-title">📋 Summary Statistics</div>', unsafe_allow_html=True)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Data Points", f"{hist_data['total_points']:,}")
        sc2.metric("Price Range", f"${df['close'].min():,.2f} – ${df['close'].max():,.2f}")
        sc3.metric("Avg Sentiment", f"{df['sentiment'].mean():.4f}")
        sc4.metric("Sentiment Std", f"{df['sentiment'].std():.4f}")

        # ── Raw Data Expander ────────────────────────────────────
        with st.expander("📄 Raw Data Table"):
            display_df = df.copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            display_df.columns = ["Timestamp", "Close ($)", "Volume", "Sentiment"]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
            csv = display_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                file_name=f"{ticker}_sentiprice_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    else:
        st.info("No historical data available for the selected ticker and date range.")

else:
    # ── Welcome State ────────────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
        <h2 style="color: #ffffff; font-weight: 600; margin-bottom: 0.5rem;">
            Ready to Analyze
        </h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 1.05rem; max-width: 500px; margin: 0 auto;">
            Enter a ticker symbol in the sidebar and click <strong>⚡ Analyze</strong> 
            to get real-time price predictions powered by BERT sentiment analysis 
            and LSTM time-series forecasting.
        </p>
        <br>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">📰</div>
                <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.3rem;">
                    BERT Sentiment
                </div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">📈</div>
                <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.3rem;">
                    LSTM Forecasting
                </div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">⚡</div>
                <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.3rem;">
                    Real-Time API
                </div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">🔀</div>
                <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.3rem;">
                    Multi-Modal Fusion
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)