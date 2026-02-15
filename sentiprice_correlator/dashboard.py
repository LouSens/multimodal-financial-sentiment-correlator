"""
dashboard.py — Premium Streamlit Dashboard for Sentiprice Correlator

Features
────────
  • Polished dark-themed UI with glassmorphism cards
  • Persistent sidebar: API health, ticker controls, model info, tuning results
  • 4 animated KPI metric cards (price, prediction, change, sentiment)
  • Dual-axis chart: Price + Sentiment overlay
  • Volume bar chart
  • Summary statistics and downloadable raw data
  • Full API integration (/health, /predict, /history, /model/info, /models)

Run
───
  streamlit run dashboard.py
"""

import os
import json
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

def is_market_open() -> bool:
    """Simple check for US Market Hours (Mon-Fri 9:30-16:00 ET)."""
    # ET is roughly UTC-5 (standard) or UTC-4 (daylight)
    # For simplicity/robustness without heavy libs, we'll use UTC-5
    now_utc = datetime.now(timezone.utc)
    
    # Check Weekend
    if now_utc.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
        
    # Convert to roughly ET (UTC-5)
    # hour in 24h format
    hour_et = (now_utc.hour - 5) % 24
    minute = now_utc.minute
    
    # Market hours: 09:30 to 16:00
    current_time = hour_et * 60 + minute
    start_time = 9 * 60 + 30
    end_time = 16 * 60
    
    return start_time <= current_time < end_time

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.join(BASE_DIR, "tuning_results")

# ===================================================================== #
#  Custom CSS — Dark Theme + Glassmorphism (Streamlit 1.32+ compatible)
# ===================================================================== #
st.markdown("""
<style>
    /* ── Import Google Font + Material Symbols ──────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* ── Fix Material Icons rendering as text ──────────────────── */
    .material-symbols-rounded,
    [data-testid="stSidebar"] button[kind="header"] span,
    .st-emotion-cache-1pbsqtx {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 24px !important;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Hide sidebar collapse arrow text fallback ─────────────── */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        font-size: 0 !important;
        color: transparent !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        font-size: 24px !important;
        color: rgba(255, 255, 255, 0.6) !important;
    }

    /* ── Global Font (Exclude Material Icons) ───────────────────── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Apply Inter to all elements EXCEPT those that use icon fonts */
    .stApp > header, .stApp > .main .block-container, .stApp > footer {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Ensure Material Icons use the correct font family */
    .material-symbols-rounded, 
    .material-icons,
    [data-testid="stSidebar"] button[kind="header"] span,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] svg,
    .st-emotion-cache-1pbsqtx,
    i {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal;
        font-style: normal;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Custom HR divider (replaces st.divider) ────────────────── */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 0.8rem 0;
    }

    /* ── Header Gradient Bar ────────────────────────────────────── */
    .header-bar {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .header-bar h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-bar p {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.9rem;
        margin-top: 0.35rem;
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
        padding: 1.3rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(79, 70, 229, 0.15);
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 500;
    }
    .delta-positive { color: #34d399; }
    .delta-negative { color: #f87171; }
    .delta-neutral  { color: #fbbf24; }
    .color-white { color: #ffffff; }
    .color-green { color: #34d399; }
    .color-red   { color: #f87171; }
    .color-blue  { color: #60a5fa; }
    .color-amber { color: #fbbf24; }
    .color-purple { color: #a78bfa; }

    /* ── Status Dot ─────────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    .status-online { background: #34d399; box-shadow: 0 0 8px #34d399; }
    .status-offline { background: #f87171; box-shadow: 0 0 8px #f87171; }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Section Title ──────────────────────────────────────────── */
    .section-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Sidebar Styling ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.98) 0%, rgba(36, 36, 62, 0.98) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* ── Sidebar Section Box ──────────────────────────────────── */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-section-title {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* ── Tuning Result Badges ──────────────────────────────────── */
    .tuning-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .tuning-stat:last-child { border-bottom: none; }
    .tuning-stat-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.78rem;
    }
    .tuning-stat-value {
        color: #ffffff;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ── Welcome Features Grid ─────────────────────────────────── */
    .features-grid {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }
    .feature-item {
        text-align: center;
        transition: transform 0.2s ease;
    }
    .feature-item:hover { transform: translateY(-3px); }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .feature-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ── Plotly chart container ──────────────────────────────────── */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.72rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }

    /* ── Hide default Streamlit elements (safe subset) ──────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Fix Streamlit expander arrow icon rendering as text ────── */
    [data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif !important;
    }
    details[data-testid="stExpander"] > summary::before,
    details[data-testid="stExpander"] > summary::after {
        font-family: 'Material Symbols Rounded' !important;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================== #
#  API Helpers
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


def load_tuning_results(ticker: str):
    """Load the latest Optuna tuning results for a ticker from disk."""
    path = os.path.join(TUNING_DIR, f"{ticker}_tuning.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ===================================================================== #
#  Sidebar
# ===================================================================== #
with st.sidebar:
    # ── Logo / Brand ─────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🧠</div>
        <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700; letter-spacing: -0.3px;">
            Sentiprice
        </div>
        <div style="color: rgba(255,255,255,0.35); font-size: 0.7rem; font-weight: 400;">
            AI Financial Correlator
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Backend Health ──────────────────────────────────────────
    health, health_err = api_get("/health")

    st.markdown('<div class="sidebar-section-title">⚡ System Status</div>', unsafe_allow_html=True)

    if health:
        status = health["status"]
        dot_cls = "status-online" if status == "healthy" else "status-offline"
        st.markdown(
            f'<span class="status-dot {dot_cls}"></span> '
            f'**API** &nbsp;·&nbsp; {status.capitalize()} &nbsp;·&nbsp; `{health["device"]}`',
            unsafe_allow_html=True,
        )
        uptime = health.get("uptime_seconds", 0)
        if uptime > 3600:
            uptime_str = f"{uptime / 3600:.1f}h"
        elif uptime > 60:
            uptime_str = f"{uptime / 60:.0f}m"
        else:
            uptime_str = f"{uptime:.0f}s"
        st.caption(f"Uptime: {uptime_str}")

        avail = health.get("available_models", [])
        if avail:
            st.caption(f"🏷️ Trained models: **{', '.join(avail)}**")
    else:
        st.markdown(
            '<span class="status-dot status-offline"></span> **API** &nbsp;·&nbsp; Offline',
            unsafe_allow_html=True,
        )
        st.caption("Start with: `python main.py`")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">🎛️ Controls</div>', unsafe_allow_html=True)

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        key="ticker_input",
        placeholder="e.g. AAPL, TSLA, MSFT",
    ).upper()

    days = st.slider(
        "History (days)",
        min_value=2,
        max_value=60,
        value=14,
        key="days_slider",
    )

    analyze_btn = st.button(
        "⚡ Analyze & Predict",
        use_container_width=True,
        type="primary",
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Model Info ───────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">🔧 Model Architecture</div>', unsafe_allow_html=True)

    show_model = st.checkbox("Show Model Details", value=False, key="model_toggle")
    if show_model:
        model_info, info_err = api_get("/model/info", params={"ticker": ticker})
        if model_info:
            cfg = model_info.get("training_config", {})
            st.markdown(f"""
            <div class="sidebar-section">
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Architecture</span>
                    <span class="tuning-stat-value">{model_info['architecture']}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Parameters</span>
                    <span class="tuning-stat-value">{model_info['total_parameters']:,}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Features</span>
                    <span class="tuning-stat-value">{len(model_info.get('feature_names', []))}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Hidden Dim</span>
                    <span class="tuning-stat-value">{cfg.get('hidden_dim', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Layers</span>
                    <span class="tuning-stat-value">{cfg.get('num_layers', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Learning Rate</span>
                    <span class="tuning-stat-value">{cfg.get('lr', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Seq Length</span>
                    <span class="tuning-stat-value">{cfg.get('seq_length', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Dropout</span>
                    <span class="tuning-stat-value">{cfg.get('dropout', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Trained Ticker</span>
                    <span class="tuning-stat-value">{cfg.get('ticker', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif info_err and "404" in str(info_err):
            st.warning(f"No model for {ticker}")
            st.caption(f"Train: `python train.py --ticker {ticker}`")
        else:
            st.caption("Model info unavailable (API offline?)")

    # ── Tuning Results ──────────────────────────────────────────
    tuning = load_tuning_results(ticker)
    if tuning:
        st.markdown('<div class="sidebar-section-title">🔬 Optuna Tuning Results</div>', unsafe_allow_html=True)
        show_tuning = st.checkbox("Show Tuning Details", value=False, key="tuning_toggle")
        if show_tuning:
            bp = tuning.get("best_params", {})
            beats_icon = "✅" if tuning.get("beats_naive") else "❌"
            r2 = tuning.get("best_r2", "N/A")
            dollar_mae = tuning.get("best_dollar_mae", "N/A")
            dir_acc = tuning.get("best_directional_accuracy", "N/A")

            st.markdown(f"""
            <div class="sidebar-section">
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Trials</span>
                    <span class="tuning-stat-value">{tuning.get('n_trials', 'N/A')}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Best Test MSE</span>
                    <span class="tuning-stat-value">{tuning.get('best_test_mse', 0):.5f}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Dollar MAE</span>
                    <span class="tuning-stat-value">${dollar_mae}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Dir. Accuracy</span>
                    <span class="tuning-stat-value">{dir_acc}%</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">R² Score</span>
                    <span class="tuning-stat-value">{r2}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Beats Naive?</span>
                    <span class="tuning-stat-value">{beats_icon}</span>
                </div>
                <div class="tuning-stat">
                    <span class="tuning-stat-label">Duration</span>
                    <span class="tuning-stat-value">{tuning.get('elapsed_seconds', 0):.0f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Best Hyperparameters:")
            st.code(json.dumps(bp, indent=2), language="json")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.2); font-size: 0.68rem; padding: 0.5rem 0;">
        v1.0.0 · Streamlit + FastAPI + PyTorch
    </div>
    """, unsafe_allow_html=True)


# ===================================================================== #
#  Header
# ===================================================================== #
st.markdown("""
<div class="header-bar">
    <h1>🤖 Multi-Modal Sentiment & Price Correlator</h1>
    <p>BERT-powered news sentiment analysis fused with LSTM time-series prediction · Optuna-optimized</p>
</div>
""", unsafe_allow_html=True)


# ===================================================================== #
#  Market Scanner Section
# ===================================================================== #
st.markdown('<div class="section-title">📡 Market Sentiment Scanner</div>', unsafe_allow_html=True)

with st.expander("Explore Real-Time Market Sentiment", expanded=False):
    st.caption("Scans the latest news (Headlines + Summaries) for your watchlist.")
    
    # Scanner Inputs
    col_scan1, col_scan2 = st.columns([3, 1])
    with col_scan1:
        default_tickers = "AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, META, GME, AMC, SPY"
        watchlist = st.text_input("Watchlist (comma separated)", value=default_tickers)
    
    with col_scan2:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        scan_btn = st.button("🚀 Scan Market", use_container_width=True)

    if scan_btn:
        tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
        
        with st.spinner(f"Scanning {len(tickers)} tickers via API..."):
            # Call API instead of local import
            scan_data, scan_err = api_post("/scan_market", {"tickers": tickers})
            
        if scan_err:
            st.error(f"⚠️ Scanner Error: {scan_err}")
        elif scan_data and scan_data.get("results"):
            scan_df = pd.DataFrame(scan_data["results"])
            
            # 1. Bar Chart of Sentiment
            fig_scan = go.Figure()
            colors = [
                "#34d399" if s >= 0.3 else "#f87171" if s <= -0.3 else "#fbbf24"
                for s in scan_df["Sentiment Score"]
            ]
            fig_scan.add_trace(go.Bar(
                x=scan_df["Ticker"],
                y=scan_df["Sentiment Score"],
                marker_color=colors,
                text=scan_df["Sentiment Score"].round(2),
                textposition='auto',
            ))
            fig_scan.update_layout(
                title="Sentiment Ranking",
                template="plotly_dark",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
                yaxis=dict(range=[-1.1, 1.1])
            )
            st.plotly_chart(fig_scan, use_container_width=True)
            
            # 2. Detailed Table with Links
            st.markdown("### 📰 Sentiment & Fundamentals")
            
            for _, row in scan_df.iterrows():
                # Card-like layout for each ticker
                score = row['Sentiment Score']
                color = "#34d399" if score >= 0.3 else "#f87171" if score <= -0.3 else "#fbbf24"
                
                with st.container():
                    # Top Row: Header + Score
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.03); 
                        border-left: 4px solid {color};
                        padding: 1rem; 
                        border-radius: 8px;
                        margin-bottom: 0.8rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                            <h3 style="margin:0; color:#fff;">{row['Ticker']} <span style="font-size:0.8em; opacity:0.7;">({row['Sentiment Label']})</span></h3>
                            <span style="font-weight:bold; color:{color}; font-size:1.1em;">{score:.2f}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Middle Row: Fundamentals Grid
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("P/E Ratio", f"{row.get('P/E Ratio', 'N/A')}")
                    with c2: st.metric("ROE", f"{row.get('ROE', 'N/A')}")
                    with c3: st.metric("Margin", f"{row.get('Profit Margin', 'N/A')}")
                    with c4: st.metric("Analyst", f"{row.get('Recommendation', 'N/A')}")
                    
                    # Bottom Row: News
                    st.markdown(f"""
                        <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);">
                        <p style="margin: 0.5rem 0; font-size: 0.95rem; font-weight: 500; color: #e2e8f0;">{row['Top Headline']}</p>
                        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8;">{row['Top Summary']}</p>
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #64748b;">
                            Source: {row['Provider']} • <a href="{row['URL']}" target="_blank" style="color: #60a5fa; text-decoration: none;">Read Full Story ↗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.warning("No recent news found for these tickers.")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ===================================================================== #
#  Main Content
# ===================================================================== #
if analyze_btn:
    # ── Prediction ──────────────────────────────────────────────
    with st.spinner(f"🔮 Analyzing {ticker}..."):
        pred_data, pred_err = api_post("/predict", {"ticker": ticker, "days": days})

    if pred_err:
        st.error(f"⚠️ Prediction Error: {pred_err}")
    elif pred_data:
        # ── Model Used Indicator ─────────────────────────────────
        model_used = pred_data.get("model_ticker", ticker)
        if model_used != ticker:
            st.info(f"ℹ️ Using **{model_used}** model for {ticker} (no {ticker}-specific model trained)")

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
            
            # Label logic based on market hours
            lbl = "Predicted (1hr)" if is_market_open() else "Predicted (Next Open)"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value {val_color}">${pred_data['predicted_next_hour']:,.2f}</div>
                <div class="metric-delta {chg_cls}">{arrow} ${abs(chg):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            pct = pred_data["price_change_pct"]
            pct_cls = delta_class(pct)
            pct_arrow = "▲" if pct > 0 else "▼" if pct < 0 else "●"
            pct_color = "color-green" if pct > 0 else "color-red" if pct < 0 else "color-amber"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price Change</div>
                <div class="metric-value {pct_color}">{pct_arrow} {abs(pct):.4f}%</div>
                <div class="metric-delta delta-neutral">Next Hour</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            sent = pred_data["sentiment_score"]
            has_news = pred_data.get("has_news", True)  # Default to True for backward compat if missing

            if not has_news:
                # ── No News State ──────────────────
                st.markdown(f"""
                <div class="metric-card" style="border-color: rgba(255,255,255,0.05); background: rgba(255,255,255,0.02);">
                    <div class="metric-label">Sentiment</div>
                    <div class="metric-value color-white" style="font-size: 1.4rem; opacity: 0.6;">No News</div>
                    <div class="metric-delta" style="color: rgba(255,255,255,0.3);">Last 48h</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── Normal State ──────────────────
                sent_cls = sentiment_color(sent)
                sent_label = sentiment_emoji(sent)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sentiment</div>
                    <div class="metric-value {sent_cls}">{sent:.4f}</div>
                    <div class="metric-delta delta-neutral">{sent_label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Extra Spacing ────────────────────────────────────────
        st.markdown("<br><br>", unsafe_allow_html=True)
            
        # ── Fundamentals Section (New) ───────────────────────────
        fund = pred_data.get("fundamentals", {})
        if fund:
            st.markdown('<div class="section-title">🏥 Fundamental Health</div>', unsafe_allow_html=True)
            f1, f2, f3, f4, f5 = st.columns(5)
            
            def safe_fmt(val, is_pct=False, prefix=""):
                if val is None: return "N/A"
                if is_pct: return f"{val*100:.1f}%"
                return f"{prefix}{val:,.2f}"

            def fmt_mcap(val):
                if not val: return "N/A"
                if val >= 1e12: return f"{val/1e12:.2f}T"
                if val >= 1e9: return f"{val/1e9:.2f}B"
                if val >= 1e6: return f"{val/1e6:.2f}M"
                return f"{val:,.0f}"

            with f1: st.metric("Market Cap", f"${fmt_mcap(fund.get('market_cap'))}", help="Total value of all shares")
            with f2: st.metric("P/E Ratio", safe_fmt(fund.get('trailing_pe')), help="Price to Earnings Ratio")
            with f3: st.metric("ROE", safe_fmt(fund.get('roe'), is_pct=True), help="Return on Equity")
            with f4: st.metric("Profit Margin", safe_fmt(fund.get('profit_margins'), is_pct=True))
            with f5: st.metric("Analyst Rec", str(fund.get('recommendation_key', 'N/A')).replace("_", " ").title())
            
            st.markdown("<br>", unsafe_allow_html=True)

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

        # Sentiment markers (color-coded) - SPARSE MODE
        # Only plot points that actually have news
        if "has_news" in df.columns:
            sent_df = df[df["has_news"] == True].copy()
        else:
            # Fallback if API hasn't updated or column missing
            sent_df = df[df["sentiment"] != 0.0].copy()

        colors = [
            "#34d399" if s >= 0.3 else "#f87171" if s <= -0.3 else "#fbbf24"
            for s in sent_df["sentiment"]
        ]
        
        fig.add_trace(
            go.Scatter(
                x=sent_df["timestamp"],
                y=sent_df["sentiment"],
                name="Sentiment (News)",
                mode="markers", # removed lines to show distinct news events
                marker=dict(
                    color=colors, 
                    size=8, 
                    line=dict(width=1, color="rgba(255,255,255,0.5)")
                ),
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
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", showgrid=True)
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
        vol_colors.insert(0, "#60a5fa")

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
        # ── Raw Data Table ──────────────────────────────────────
        with st.expander("📄 Raw Data Table"):
            # Explicitly select columns to match names (exclude metadata like 'has_news')
            display_df = df[["timestamp", "close", "volume", "sentiment"]].copy()
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
    <div class="glass-card" style="text-align: center; padding: 3.5rem 2rem;">
        <div style="font-size: 3.5rem; margin-bottom: 0.8rem;">🧠</div>
        <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 0.5rem; font-size: 1.6rem;">
            Ready to Analyze
        </h2>
        <p style="color: rgba(255,255,255,0.45); font-size: 0.95rem; max-width: 520px; margin: 0 auto; line-height: 1.6;">
            Enter a ticker symbol in the sidebar and click <strong style="color: rgba(255,255,255,0.7);">⚡ Analyze & Predict</strong>
            to get next-hour price predictions powered by BERT sentiment analysis
            and LSTM time-series forecasting.
        </p>
        <div class="features-grid">
            <div class="feature-item">
                <div class="feature-icon">📰</div>
                <div class="feature-label">BERT Sentiment</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-label">LSTM Forecasting</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔬</div>
                <div class="feature-label">Optuna Tuned</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-label">Real-Time API</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔀</div>
                <div class="feature-label">Multi-Modal Fusion</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-start cards ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown("""
        <div class="glass-card" style="min-height: 160px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">🏋️ Train</div>
            <p style="color: rgba(255,255,255,0.4); font-size: 0.82rem; line-height: 1.5; margin: 0;">
                Train a model for any ticker symbol using price data and news sentiment.
            </p>
            <code style="font-size: 0.72rem; color: #a78bfa; margin-top: 0.5rem; display: block;">
                python train.py --ticker AAPL
            </code>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div class="glass-card" style="min-height: 160px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">🔬 Tune</div>
            <p style="color: rgba(255,255,255,0.4); font-size: 0.82rem; line-height: 1.5; margin: 0;">
                Optimize hyperparameters with Optuna to find the best model configuration.
            </p>
            <code style="font-size: 0.72rem; color: #a78bfa; margin-top: 0.5rem; display: block;">
                python evaluate.py --ticker AAPL --trials 20
            </code>
        </div>
        """, unsafe_allow_html=True)

    with q3:
        st.markdown("""
        <div class="glass-card" style="min-height: 160px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.5rem;">🚀 Serve</div>
            <p style="color: rgba(255,255,255,0.4); font-size: 0.82rem; line-height: 1.5; margin: 0;">
                Launch the FastAPI backend and make live predictions via this dashboard.
            </p>
            <code style="font-size: 0.72rem; color: #a78bfa; margin-top: 0.5rem; display: block;">
                python main.py
            </code>
        </div>
        """, unsafe_allow_html=True)


# ── App Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Sentiprice Correlator · Built with PyTorch, FastAPI, Streamlit, Optuna · 2026
</div>
""", unsafe_allow_html=True)