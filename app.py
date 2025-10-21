# app.py
# Streamlit dashboard: Portfolio Y Signal Tracker
# Author: ChatGPT (for Meri)
# Date: 2025-10-21
#
# What this app does
# - Shows forward-looking expected return ranges for SPYI, SMH, DFNS, GOLD (proxy GLD), EQQQ
# - Uses live market signals (yfinance) to tilt a conservative base model across 4 horizons:
#   Short (0–6m), Mid (6–12m), Longer (12–36m), Decade (36–120m)
# - Signals: 10y yield, VIX, USD DXY, Gold spot, ETF momentum, NASDAQ momentum
# - You can tweak weights and see the impact in real-time
#
# How to run
#   1) pip install -r requirements.txt
#   2) streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import pytz

# -----------------------------
# Config & Constants
# -----------------------------
st.set_page_config(page_title="Portfolio Y — Signal Tracker", page_icon="📊", layout="wide")

EU_TZ = pytz.timezone("Europe/Amsterdam")

ETF_MAP = {
    "SPYI": {"yf": "SPYI", "name": "NEOS S&P 500 High Income", "category": "Covered-call US Equities"},
    "SMH":  {"yf": "SMH",  "name": "VanEck Semiconductors", "category": "Semiconductors"},
    # DFNS ticker varies by venue. Default to LSE (DFNS.L). If you prefer US listing, change to appropriate ticker.
    "DFNS": {"yf": "DFNS.L", "name": "Defense & Aerospace (DFNS)", "category": "Defense & Aerospace"},
    # GOLD proxy via GLD (SPDR Gold Shares)
    "GOLD": {"yf": "GLD",  "name": "Gold (GLD proxy)", "category": "Commodity (Gold)"},
    # EQQQ UCITS on Euronext Amsterdam
    "EQQQ": {"yf": "EQQQ.AS", "name": "Invesco Nasdaq-100 (EUR)", "category": "US Tech / Nasdaq-100"},
}

INDEX_MAP = {
    "DXY": "DX-Y.NYB",   # US Dollar Index (ICE)
    "VIX": "^VIX",
    "UST10Y_YIELD": "^TNX",  # CBOE 10Y yield *10 (e.g., 45.00 = 4.5%)
    "NASDAQ100": "^NDX",
    "SOX": "^SOX",      # Philadelphia Semiconductor Index
    "GOLD_FUT": "GC=F", # Gold futures front month
}

# Conservative base-case expected ranges (cumulative) for horizons (can be tuned in sidebar)
BASE_EXPECTED = {
    "SPYI":  {"short": (0.02, 0.04), "mid": (0.05, 0.08), "longer": (0.12, 0.20), "decade": (0.30, 0.40)},
    "SMH":   {"short": (0.08, 0.12), "mid": (0.15, 0.20), "longer": (0.35, 0.50), "decade": (0.80, 1.20)},
    "DFNS":  {"short": (0.04, 0.07), "mid": (0.09, 0.14), "longer": (0.20, 0.35), "decade": (0.50, 0.90)},
    "GOLD":  {"short": (-0.02, 0.02), "mid": (0.03, 0.07), "longer": (0.10, 0.15), "decade": (0.20, 0.30)},
    "EQQQ":  {"short": (0.06, 0.10), "mid": (0.12, 0.18), "longer": (0.25, 0.40), "decade": (0.70, 1.00)},
}

# Signal sensitivities (tilt multipliers). Positive means bullish when signal falls (e.g., lower yield helpful).
SENSITIVITIES = {
    "SPYI":  {"UST10Y": -0.6, "VIX": 0.2,  "DXY": -0.2, "NDX": 0.3, "SOX": 0.1, "GOLD": -0.1},
    "SMH":   {"UST10Y": -0.3, "VIX": -0.1, "DXY": -0.2, "NDX": 0.4, "SOX": 0.6, "GOLD": 0.0},
    "DFNS":  {"UST10Y": 0.1,  "VIX": 0.2,  "DXY": 0.1,  "NDX": 0.0, "SOX": 0.0, "GOLD": 0.1},
    "GOLD":  {"UST10Y": -0.7, "VIX": 0.2,  "DXY": -0.6, "NDX": -0.1,"SOX": -0.1,"GOLD": 0.5},
    "EQQQ":  {"UST10Y": -0.7, "VIX": -0.2, "DXY": -0.2, "NDX": 0.6, "SOX": 0.2, "GOLD": 0.0},
}

# Horizons
HORIZONS = [
    ("short", "0–6 mo"),
    ("mid", "6–12 mo"),
    ("longer", "12–36 mo"),
    ("decade", "36–120 mo"),
]

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=300)
def fetch_prices(symbols, period="1y", interval="1d"):
    df = yf.download(symbols, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
    if isinstance(symbols, list) and len(symbols) > 1:
        px = df["Close"].copy()
    else:
        px = df["Close"].to_frame()
    return px.dropna(how="all")

def pct_change_over_days(series: pd.Series, days: int) -> float:
    """Percent change over last 'days' trading days (approx)."""
    if series is None or series.empty:
        return np.nan
    end = series.dropna().iloc[-1]
    start_idx = max(0, len(series) - days)
    start = series.dropna().iloc[start_idx]
    if pd.isna(start) or start == 0:
        return np.nan
    return float((end / start) - 1.0)

def normalize_signal(value: float, reference: float) -> float:
    """Simple normalization to a z-like score (bounded)."""
    if reference == 0 or np.isnan(value) or np.isnan(reference):
        return 0.0
    z = (value - reference) / (abs(reference) + 1e-9)
    return float(np.clip(z, -1.5, 1.5))

def format_range(low, high):
    return f"{int(round(low*100))}% – {int(round(high*100))}%"

def color_for_range(low, high):
    mid = (low+high)/2
    if mid >= 0.12:
        return "🟢"
    if mid >= 0.03:
        return "🟡"
    return "🟠" if mid >= 0 else "🔴"

# -----------------------------
# Sidebar — controls
# -----------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.caption("Tune base expectations and signal weights.")

base_scale = st.sidebar.slider("Base expectation scale", 0.5, 1.5, 1.0, 0.05,
                               help="Multiplies all base expected ranges before signal tilts.")

signal_strength = st.sidebar.slider("Signal tilt strength", 0.0, 2.0, 1.0, 0.05,
                                    help="Amplifies how strongly signals adjust the base ranges.")

st.sidebar.markdown("---")
st.sidebar.write("**DFNS ticker venue**")
dfns_choice = st.sidebar.selectbox("DFNS data source", ["DFNS.L (LSE)", "Custom..."], index=0)
if dfns_choice == "Custom...":
    user_dfns = st.sidebar.text_input("Enter custom yfinance ticker for DFNS-like exposure", value="")
    if user_dfns.strip():
        ETF_MAP["DFNS"]["yf"] = user_dfns.strip()

st.sidebar.markdown("---")
st.sidebar.write("**Model Notes**")
show_notes = st.sidebar.checkbox("Show calculation notes per ETF", value=True)

# -----------------------------
# Fetch live data
# -----------------------------
with st.spinner("Fetching market data..."):
    etf_symbols = [cfg["yf"] for cfg in ETF_MAP.values()]
    etf_prices = fetch_prices(etf_symbols, period="2y", interval="1d")

    idx_symbols = list(INDEX_MAP.values())
    idx_prices = fetch_prices(idx_symbols, period="2y", interval="1d")

# Current signal snapshot
def get_last(symbol, frame):
    try:
        return float(frame[symbol].dropna().iloc[-1])
    except Exception:
        return np.nan

signal_snapshot = {
    "VIX": get_last(INDEX_MAP["VIX"], idx_prices),
    "UST10Y": get_last(INDEX_MAP["UST10Y_YIELD"], idx_prices)/10.0 if not np.isnan(get_last(INDEX_MAP["UST10Y_YIELD"], idx_prices)) else np.nan,  # convert ^TNX
    "DXY": get_last(INDEX_MAP["DXY"], idx_prices),
    "NDX": get_last(INDEX_MAP["NASDAQ100"], idx_prices),
    "SOX": get_last(INDEX_MAP["SOX"], idx_prices),
    "GOLD_FUT": get_last(INDEX_MAP["GOLD_FUT"], idx_prices),
}

# Reference anchors (12m medians as baseline)
def rolling_median(series: pd.Series, window=252):
    if series is None or series.empty:
        return np.nan
    return float(series.dropna().tail(window).median())

reference_levels = {
    "VIX": rolling_median(idx_prices[INDEX_MAP["VIX"]]),
    "UST10Y": rolling_median(idx_prices[INDEX_MAP["UST10Y_YIELD"]])/10.0 if INDEX_MAP["UST10Y_YIELD"] in idx_prices else np.nan,
    "DXY": rolling_median(idx_prices[INDEX_MAP["DXY"]]),
    "NDX": rolling_median(idx_prices[INDEX_MAP["NASDAQ100"]]),
    "SOX": rolling_median(idx_prices[INDEX_MAP["SOX"]]),
    "GOLD_FUT": rolling_median(idx_prices[INDEX_MAP["GOLD_FUT"]]),
}

# -----------------------------
# Model: adjust base ranges by signals
# -----------------------------
def adjust_ranges_for_signals(etf_key: str, base_ranges: dict, snap: dict, refs: dict,
                              sens: dict, base_scale: float, signal_strength: float):
    out = {}
    # Build a composite tilt from normalized deviations times sensitivities
    components = []
    for key, coef in sens.items():
        if key == "UST10Y":
            s_val = snap["UST10Y"]
            r_val = refs["UST10Y"]
        elif key == "VIX":
            s_val = snap["VIX"]; r_val = refs["VIX"]
        elif key == "DXY":
            s_val = snap["DXY"]; r_val = refs["DXY"]
        elif key == "NDX":
            s_val = snap["NDX"]; r_val = refs["NDX"]
        elif key == "SOX":
            s_val = snap["SOX"]; r_val = refs["SOX"]
        elif key == "GOLD":
            s_val = snap["GOLD_FUT"]; r_val = refs["GOLD_FUT"]
        else:
            continue
        z = normalize_signal(s_val, r_val)
        components.append(coef * z)

    composite_tilt = float(np.tanh(signal_strength * np.sum(components)))  # squash to [-1,1]

    for hz, rng in base_ranges.items():
        low, high = rng
        # Scale base first
        low *= base_scale; high *= base_scale
        # Apply tilt: tilt expands/contracts symmetrically by up to 30% of the base width
        width = high - low
        tilt_factor = 0.30 * composite_tilt  # up to ±30% width
        low_adj = low + (width * tilt_factor * 0.25)   # smaller effect on lower bound
        high_adj = high + (width * tilt_factor)
        out[hz] = (max(-0.5, low_adj), min(2.0, high_adj))  # cap for sanity
    return out, composite_tilt, components

# -----------------------------
# UI Header
# -----------------------------
st.title("📊 Portfolio Y — Live Signal Tracker")
st.caption("Dynamic horizon forecasts for SPYI, SMH, DFNS, GOLD (GLD proxy), and EQQQ.")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("VIX (volatility)", f"{signal_snapshot['VIX']:.2f}")
with colB:
    st.metric("US 10Y Yield", f"{signal_snapshot['UST10Y']:.2f}%")
with colC:
    st.metric("DXY (US Dollar Index)", f"{signal_snapshot['DXY']:.2f}")
with colD:
    st.metric("Gold (front)", f"{signal_snapshot['GOLD_FUT']:.0f}")

st.caption(f"Last updated: {datetime.now(EU_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")

st.markdown("---")

# -----------------------------
# Main Table
# -----------------------------
rows = []
notes = []

for etf_key, cfg in ETF_MAP.items():
    base_rngs = BASE_EXPECTED[etf_key]
    adj, tilt, comps = adjust_ranges_for_signals(
        etf_key, base_rngs, signal_snapshot, reference_levels, SENSITIVITIES[etf_key],
        base_scale=base_scale, signal_strength=signal_strength
    )

    row = {
        "Ticker": etf_key,
        "Name": cfg["name"],
        "Category": cfg["category"],
    }
    for hz, label in HORIZONS:
        low, high = adj[hz]
        row[f"{label}"] = f"{color_for_range(low, high)} {format_range(low, high)}"
    rows.append(row)

    if show_notes:
        comps_named = {k: round(v,3) for k,v in zip(SENSITIVITIES[etf_key].keys(), comps)}
        notes.append({
            "Ticker": etf_key,
            "Composite tilt (−1…+1)": round(tilt, 3),
            "Signal contributions": comps_named
        })

table_df = pd.DataFrame(rows)
st.subheader("Forecast by Horizon (cumulative %)")
st.dataframe(table_df, use_container_width=True)

if show_notes:
    st.subheader("Calculation Notes")
    for n in notes:
        with st.expander(f"Notes for {n['Ticker']}"):
            st.write(f"**Composite tilt**: {n['Composite tilt (−1…+1)']}")
            st.json(n["Signal contributions"])

st.markdown("---")
st.caption("This dashboard uses a conservative, mean-reverting base model with signal-driven tilts. "
           "Ranges are cumulative returns for each horizon, not annualised.")
