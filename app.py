import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
import math
import time

st.set_page_config(page_title="BTC Fair-Value Scanner", layout="centered")

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
EVENT_START_HOUR_EST = 9  # first event at 9am EST
EVENT_END_HOUR_EST = 17   # last event at 5pm EST (5pm)
STRIKE_SPACING = 250       # dollar spacing
NUM_STRIKES_EACH_SIDE = 10 # number of strikes above/below ATM

# ----------------------------------------------------------
# AUTO-REFRESH TOGGLE
# ----------------------------------------------------------
refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)

if refresh:
    st.experimental_set_query_params(ts=int(time.time()))
    time.sleep(5)
    st.experimental_rerun()

# ----------------------------------------------------------
# FETCH BTC PRICE (KRAKEN INDEX – CF STYLE)
# ----------------------------------------------------------
def fetch_btc_spot():
    url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        key = list(data["result"].keys())[0]
        price = float(data["result"][key]["c"][0])
        return price
    except:
        return None

# ----------------------------------------------------------
# FETCH VOLATILITY (KRAKEN 1-MIN OHLC)
# ----------------------------------------------------------
def fetch_kraken_ohlc(n_minutes=240):
    url = f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        raw = r.json()["result"]
        key = list(raw.keys())[0]
        df = pd.DataFrame(raw[key], columns=[
            "time","open","high","low","close","vwap","volume","count"
        ])
        df["close"] = df["close"].astype(float)
        df = df.tail(n_minutes)
        return df
    except:
        return None

def compute_sigma_per_minute(df):
    closes = df["close"].values
    if len(closes) < 2:
        return 0.0008
    logs = np.log(closes[1:] / closes[:-1])
    sigma = np.std(logs, ddof=1)
    return max(sigma, 1e-6)

# ----------------------------------------------------------
# NEXT EVENT DETECTOR (HOURLY, STARTING 9AM EST)
# ----------------------------------------------------------
def get_next_event_time():
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_est = now_utc.astimezone(ZoneInfo("America/New_York"))

    today_est = now_est.date()
    for hour in range(EVENT_START_HOUR_EST, EVENT_END_HOUR_EST + 1):
        event_time_est = dt.datetime(
            today_est.year, today_est.month, today_est.day,
            hour, 0, 0, tzinfo=ZoneInfo("America/New_York")
        )
        if event_time_est > now_est:
            return event_time_est

    # If past last event → use tomorrow's 9am
    tomorrow = today_est + dt.timedelta(days=1)
    return dt.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day,
        EVENT_START_HOUR_EST, 0, 0,
        tzinfo=ZoneInfo("America/New_York")
    )

# ----------------------------------------------------------
# PROBABILITY CALCULATION (LOG-NORMAL MODEL)
# ----------------------------------------------------------
def prob_price_above(S0, K, sigma_per_minute, minutes_left):
    if minutes_left <= 0:
        return 1.0 if S0 > K else 0.0
    sigma_T = sigma_per_minute * math.sqrt(minutes_left)
    if sigma_T <= 0:
        return 1.0 if S0 > K else 0.0
    z = (math.log(K) - math.log(S0)) / sigma_T
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 1 - Phi

# ----------------------------------------------------------
# UI HEADER
# ----------------------------------------------------------
st.title("🔍 BTC Fair-Value Strike Scanner")
st.caption("Fully automatic • CF-style Kraken Index • iPhone optimized")

# ----------------------------------------------------------
# GET BTC SPOT + VOLATILITY
# ----------------------------------------------------------
spot = fetch_btc_spot()
if spot is None:
    st.error("Could not fetch BTC price.")
    st.stop()

df_ohlc = fetch_kraken_ohlc(240)
if df_ohlc is None:
    st.error("Could not fetch Kraken OHLC.")
    st.stop()

sigma = compute_sigma_per_minute(df_ohlc)

# ----------------------------------------------------------
# EVENT TIME + MINUTES REMAINING
# ----------------------------------------------------------
event_est = get_next_event_time()
event_utc = event_est.astimezone(dt.timezone.utc)
now_utc = dt.datetime.now(dt.timezone.utc)
minutes_left = max(0, int((event_utc - now_utc).total_seconds() // 60))

# ----------------------------------------------------------
# GENERATE STRIKE LIST AUTOMATICALLY
# ----------------------------------------------------------
atm = round(spot / STRIKE_SPACING) * STRIKE_SPACING
strikes = [atm + i * STRIKE_SPACING for i in range(-NUM_STRIKES_EACH_SIDE, NUM_STRIKES_EACH_SIDE + 1)]

# ----------------------------------------------------------
# CALCULATE FAIR VALUES
# ----------------------------------------------------------
rows = []
for K in strikes:
    win = prob_price_above(spot, K, sigma, minutes_left)
    fair_yes = win
    rows.append([K, fair_yes, win])

df = pd.DataFrame(rows, columns=["Strike", "Fair_Yes", "Win_Prob"])
df = df.sort_values("Fair_Yes", ascending=False)

# ----------------------------------------------------------
# BEST STRIKE SECTION
# ----------------------------------------------------------
best = df.iloc[0]

st.subheader("⭐ Best Strike Right Now")
st.metric("Strike", f"${best['Strike']:,}")
st.metric("Fair YES Price", f"{best['Fair_Yes']:.3f}")
st.metric("Win Probability", f"{best['Win_Prob']*100:.1f}%")
st.metric("Minutes to Settle", minutes_left)

st.markdown("---")

# ----------------------------------------------------------
# TABLE (Top 15)
# ----------------------------------------------------------
st.subheader("Top 15 Strikes (Fair Values)")
st.dataframe(df.head(15), use_container_width=True)

st.markdown("---")
st.caption("Compare Fair YES to Kalshi YES for instant opportunities.")
