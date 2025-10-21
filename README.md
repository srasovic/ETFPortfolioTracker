# Portfolio Y — Streamlit Signal Tracker

A Streamlit web app that displays **forward-looking expected return ranges** (cumulative, not annualised) for your Portfolio Y ETFs:
- SPYI, SMH, DFNS, GOLD (GLD proxy), EQQQ

## Features
- 4 horizons: Short (0–6m), Mid (6–12m), Longer (12–36m), Decade (36–120m)
- Conservative base expectations + **live market signals** tilts:
  - VIX, US 10Y yield (^TNX), DXY, Gold front future (GC=F), Nasdaq-100 (^NDX), SOX (^SOX)
- Adjustable **base scale** and **signal tilt strength**
- Notes per ETF with composite tilt and signal contributions

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Ticker Notes
- **DFNS** varies by exchange. Default is `DFNS.L` (LSE). You can change it in the sidebar to match your venue.
- **EQQQ** uses `EQQQ.AS` (Euronext Amsterdam) via yfinance.
- **GOLD** is proxied via `GLD`.

## Model Notes
- The base ranges are conservative and mean-reverting, then **tilted** by the current signal deviations (vs 12m rolling medians) using ETF-specific sensitivities.
- All horizon outputs are **cumulative returns**, not annualised.