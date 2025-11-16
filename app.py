import streamlit as st
import requests
import numpy as np
import pandas as pd
import datetime as dt
import math

st.set_page_config(page_title="Kalshi BTC Hourly EV (Mobile)", layout="centered")

# ---------------------------------------------------
# PRICE SOURCES (FREE)
# ---------------------------------------------------

KRAKEN_TICKER = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
COINGECKO_MARKET_CHART = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?"
    "vs_currency=usd&days=1&interval=minutely"
)

def fetch_btc_price_kraken_index():
    """Free CF-derived BTC index via Kraken."""
    r = requests.get(KRAKEN_TICKER, timeout=8)
    r.raise_for_status()
    data = r.json()
    result = data["result"]
    key = list(result.keys())[0]  # usually XXBTZUSD
    last_trade = result[key]["c"][0]
    return float(last_trade)

def fetch_btc_price_coingecko():
    """Fallback BTC price."""
    r = requests.get(COINGECKO_PRICE, timeout=8)
    r.raise_for_status()
    return r.json()["bitcoin"]["usd"]

def fetch_current_btc_price():
    """DEFAULT: Kraken index (CF-like). FALLBACK: CoinGecko."""
    try:
        return fetch_btc_price_kraken_index(), "Kraken Index (CF-derived)"
    except:
        try:
            price = fetch_btc_price_coingecko()
            return price, "CoinGecko (fallback)"
        except:
            return None, "Error fetching price"

def fetch_minute_prices_last_n(n_minutes=240):
    """Minute-level BTC data from CoinGecko."""
    r = requests.get(COINGECKO_MARKET_CHART, timeout=10)
    r.raise_for_status()
    raw = r.json().get("prices", [])
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts_ms", "price"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms")
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    return df.tail(n_minutes)

def realized_log_sigma_per_minute(prices_series):
    """Volatility estimate from log returns."""
    p = np.array(prices_series)
    if len(p) < 2:
        return None
    logr = np.log(p[1:] / p[:-1])
    return float(np.std(logr, ddof=1))

def prob_price_above_strike_log_normal(S0, K, sigma_log_per_minute, minutes_remaining):
    """Probability BTC ends above strike (log-normal model)."""
    if minutes_remaining <= 0:
        return 1.0 if S0 > K else 0.0

    sigma_T = sigma_log_per_minute * math.sqrt(minutes_remaining)
    if sigma_T <= 0:
        return 1.0 if S0 > K else 0.0

    z = (math.log(K) - math.log(S0)) / sigma_T
    from math import erf, sqrt
    Phi = 0.5 * (1 + erf(z / math.sqrt(2)))  
    return max(0.0, min(1.0, 1 - Phi))

def human_pct(x):
    return f"{100*x:.2f}%"


# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("Kalshi BTC Hourly – EV Helper")
st.caption("CF-derived price (Kraken) • Mobile-optimized • Free data only")

col1, col2 = st.columns([2,1])

with col1:
    strike = st.number_input(
        "Kalshi Strike (BTC above this at hour end?)",
        value=96500.0,
        step=50.0,
        format="%.2f"
    )
    minutes_for_vol = st.slider(
        "Use last N minutes to estimate volatility:",
        min_value=30,
        max_value=1440,
        value=240,
        step=30
    )
    ev_threshold = st.number_input(
        "EV threshold to recommend BUY (decimal)",
        value=0.02,
        step=0.01,
        format="%.2f"
    )

with col2:
    kalshi_price = st.number_input(
        "Kalshi YES price (0–1)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        format="%.4f"
    )

# ---------------------------------------------------
# TIME UNTIL HOURLY SETTLEMENT (FIXED)
# ---------------------------------------------------

now = dt.datetime.utcnow()
hour_start = now.replace(minute=0, second=0, microsecond=0)
hour_end = hour_start + dt.timedelta(hours=1)
minutes_remaining = int((hour_end - now).total_seconds() // 60)
minutes_remaining = max(0, minutes_remaining)

st.markdown("---")

# ---------------------------------------------------
# FETCH BTC PRICE
# ---------------------------------------------------

status_box = st.empty()
btc_price, source = fetch_current_btc_price()

if btc_price is None:
    status_box.error("Error fetching BTC price.")
else:
    status_box.info(
        f"BTC from {source}: ${btc_price:,.0f} • UTC {now.strftime('%H:%M')}"
    )

# ---------------------------------------------------
# VOLATILITY
# ---------------------------------------------------

sigma = None
try:
    df_min = fetch_minute_prices_last_n(minutes_for_vol)
    if not df_min.empty:
        sigma = realized_log_sigma_per_minute(df_min["price"].values)
except Exception as e:
    st.warning(f"Could not fetch minute price series: {e}")

if sigma is None or sigma == 0:
    sigma = 0.0008  # fallback volatility

# ---------------------------------------------------
# WIN PROBABILITY + EV
# ---------------------------------------------------

win_prob = prob_price_above_strike_log_normal(
    S0=btc_price if btc_price else strike,
    K=strike,
    sigma_log_per_minute=sigma,
    minutes_remaining=minutes_remaining
)

ev = win_prob - kalshi_price
no_win_prob = 1 - win_prob
no_price = 1 - kalshi_price
no_ev = no_win_prob - no_price

# ---------------------------------------------------
# DISPLAY SNAPSHOT (MOBILE)
# ---------------------------------------------------

st.markdown("## Live Snapshot")
cA, cB = st.columns(2)

with cA:
    st.metric("BTC price", f"${btc_price:,.2f}" if btc_price else "—")
    st.metric("Strike", f"${strike:,.2f}")
    st.metric("Minutes left", f"{minutes_remaining} min")

with cB:
    st.metric("Win prob (model)", human_pct(win_prob))
    st.metric("Implied prob", human_pct(kalshi_price))
    st.metric("EV (YES)", f"{ev:.4f}")

st.markdown("---")

# ---------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------

if ev >= ev_threshold:
    rec = "BUY YES"
    emoji = "✅"
elif no_ev >= ev_threshold:
    rec = "BUY NO"
    emoji = "🔻"
else:
    rec = "WAIT"
    emoji = "⏳"

st.markdown(f"### Recommendation: {emoji} **{rec}**")
st.write(f"- YES win probability: **{human_pct(win_prob)}**")
st.write(f"- YES price: **{kalshi_price:.3f}**")
st.write(f"- EV(YES): **{ev:.4f}**")
st.write(f"- EV(NO): **{no_ev:.4f}**")
st.write(f"- ROI if YES wins: **{(1-kalshi_price)/kalshi_price:.2f}×**")

st.markdown("---")
st.caption("Tip: Safari → Share → Add to Home Screen for app-like usage.")

if st.button("Refresh now"):
    st.experimental_rerun()
