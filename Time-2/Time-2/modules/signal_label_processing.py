import numpy as np
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from sklearn.linear_model import LinearRegression


# ============================================================
# GROUP 1: EXTREMUM-BASED ENTRY SIGNAL GENERATION (ENTRY ONLY)
# ============================================================
def generate_signal_only_extrema(df, cluster_length=30, use_hilo=True):
    out = df.copy()
    out["Signal"] = 0

    if use_hilo:
        max_idx = argrelextrema(out["high"].values, np.greater, order=cluster_length)[0]
        min_idx = argrelextrema(out["low"].values, np.less, order=cluster_length)[0]
    else:
        max_idx = argrelextrema(out["close"].values, np.greater, order=cluster_length)[0]
        min_idx = argrelextrema(out["close"].values, np.less, order=cluster_length)[0]

    out.iloc[max_idx, out.columns.get_loc("Signal")] = -1
    out.iloc[min_idx, out.columns.get_loc("Signal")] = 1
    return out


# ============================================================
# SIGNAL TRANSFORMS (FOR OPTIMIZATION / TUNING)
# ============================================================
def shift_signals(df, delay=3):
    """
    Shift entry signals forward by delay bars (optional tuning).
    Safer integer-position approach.
    """
    out = df.copy()
    sig = out["Signal"].to_numpy().copy()
    out["Signal"] = 0

    for i in range(len(sig)):
        if sig[i] != 0:
            pos = i + delay
            if pos < len(out):
                out.iat[pos, out.columns.get_loc("Signal")] = int(sig[i])

    print(f"Shift signals forward by delay={delay} bars.")
    return out


def enforce_min_distance(df, min_bars=20):
    """
    Keep only one non-zero entry signal within any min_bars window.
    """
    out = df.copy()
    sig_col = out.columns.get_loc("Signal")

    last_pos = -10**9
    for i in range(len(out)):
        if out.iat[i, sig_col] != 0:
            if i - last_pos < min_bars:
                out.iat[i, sig_col] = 0
            else:
                last_pos = i
    return out


def remove_low_volatility_signals(df, threshold_percentile=20, atr_period=14):
    """
    Nullify entry signals when ATR is below chosen percentile threshold.
    """
    out = df.copy()

    required = ["high", "low", "close", "Signal"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    atr = AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=atr_period
    ).average_true_range()

    out["ATR"] = atr  # optional: useful for debug/report
    thr = np.nanpercentile(atr.dropna().values, threshold_percentile)

    low_vol = atr < thr
    out.loc[low_vol, "Signal"] = 0

    print(f"Low-volatility threshold ATR percentile={threshold_percentile}% => {thr:.6f}")
    print(f"After ATR filter: {out['Signal'].value_counts().to_dict()}")
    return out


def filter_by_slope(df, look_ahead=24, slope_threshold=0.0):
    """
    Trend confirmation using linear regression slope.
    Apply ONLY to ENTRY signals (Signal != 0).
    BUY requires slope > +threshold
    SELL requires slope < -threshold
    """
    out = df.copy()
    print(f"Before slope filter: {out['Signal'].value_counts().to_dict()}")

    lr = LinearRegression()
    signal_idx = out.index[out["Signal"] != 0].tolist()

    for idx in signal_idx:
        pos = out.index.get_loc(idx)
        if pos + look_ahead < len(out):
            y = out["close"].iloc[pos:pos + look_ahead].values.reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)
            lr.fit(x, y)
            slope = float(lr.coef_.flatten()[0])

            sig = int(out.loc[idx, "Signal"])

            # BUY must have slope > +threshold
            if sig == 1 and slope <= slope_threshold:
                out.loc[idx, "Signal"] = 0

            # SELL must have slope < -threshold
            elif sig == -1 and slope >= -slope_threshold:
                out.loc[idx, "Signal"] = 0

    print(f"After slope filter: {out['Signal'].value_counts().to_dict()}")
    return out


def prior_signal_making_zero(df_signal, reset_length=5):
    out = df_signal.copy()
    reset_indexes = set()

    sig_col = out.columns.get_loc("Signal")

    for i in range(1, len(out)):
        cur_sig = int(out.iat[i, sig_col])
        prev_sig = int(out.iat[i - 1, sig_col])

        if (cur_sig == 1 and prev_sig == -1) or (cur_sig == -1 and prev_sig == 1):
            for j in range(i, i - reset_length - 1, -1):
                if j >= 0:
                    reset_indexes.add(j)

    if reset_indexes:
        out.iloc[list(reset_indexes), sig_col] = 0

    print(f"After nullify prior {reset_length} bars: {out['Signal'].value_counts().to_dict()}")
    return out


# ============================================================
# PLOT ONLY: POSITION-STATE PROPAGATION (DO NOT USE FOR ML)
# ============================================================
def signal_propagate_for_plot_only(df_signals):
    out = df_signals.copy()
    current = 0
    for idx in out.index:
        if out.loc[idx, "Signal"] != 0:
            current = int(out.loc[idx, "Signal"])
        out.loc[idx, "Signal"] = current

    print(f"Signals After Propagation (PLOT ONLY): {out['Signal'].value_counts().to_dict()}")
    return out


def generate_consecutive_signal_label(df, col="Signal"):
    out = df.copy()
    signal = out[col].to_numpy(dtype=int)

    propagated = np.zeros_like(signal)
    last_value = 0
    for i, val in enumerate(signal):
        if val != 0:
            last_value = val
        propagated[i] = last_value
    return propagated


# ============================================================
# STABLE ML LABEL (RETURN-BASED)
# ============================================================
def make_return_trend_label(df, horizon=5, thr=0.002, price_col="close"):
    out = df.copy()
    future_ret = out[price_col].shift(-horizon) / out[price_col] - 1.0

    out["Label"] = 0
    out.loc[future_ret > thr, "Label"] = 1
    out.loc[future_ret < -thr, "Label"] = -1
    return out
