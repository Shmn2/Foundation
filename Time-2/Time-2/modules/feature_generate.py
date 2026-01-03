import ta
import numpy as np
import pandas as pd


# ================================
# 1️⃣ Base Features (Compact + Useful)
# ================================
def extract_buy_sell_hold_features(df):
    df = df.copy()

    # ========= TREND FEATURES =========
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df["ma_alignment"] = (
        (df["sma10"] > df["sma20"]).astype(int) +
        (df["sma20"] > df["sma50"]).astype(int)
    )

    # slope (trend) - safe polyfit
    def _slope(x):
        if len(x) < 2:
            return np.nan
        try:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        except Exception:
            return np.nan

    df["slope_20"] = df["close"].rolling(20).apply(_slope, raw=True)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()

    # ========= MOMENTUM =========
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["roc"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ========= VOLATILITY =========
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()
    df["atr_n"] = df["atr"] / (df["close"] + 1e-9)

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df["close"] + 1e-9)
    df["bb_pos"] = (df["close"] - bb.bollinger_lband()) / (
        (bb.bollinger_hband() - bb.bollinger_lband()) + 1e-9
    )

    # ========= MARKET STRUCTURE =========
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = (df["high"] - df["low"]).abs()
    df["wick_ratio"] = df["candle_range"] / (df["candle_body"] + 1e-9)

    df["hh"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["ll"] = (df["low"] < df["low"].shift(1)).astype(int)

    # ========= SUPPORT & RESISTANCE (make numeric) =========
    df["pivot_high"] = df["high"].rolling(5).apply(lambda x: 1 if x[2] == np.max(x) else 0, raw=True)
    df["pivot_low"] = df["low"].rolling(5).apply(lambda x: 1 if x[2] == np.min(x) else 0, raw=True)

    df["dist_to_high"] = df["close"] - df["high"].rolling(20).max()
    df["dist_to_low"] = df["close"] - df["low"].rolling(20).min()

    # ========= REVERSAL FEATURES =========
    df["doji"] = (df["candle_body"] < df["candle_range"] * 0.1).astype(int)
    df["dir"] = np.sign(df["close"].diff())
    df["dir_change"] = df["dir"].diff().abs()

    # ========= VOLUME FEATURES =========
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_spike"] = df["volume"] / (df["volume_sma"] + 1e-9)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # ========= RISK–REWARD FEATURES =========
    df["rr_ratio"] = (df["close"] - df["low"]) / ((df["high"] - df["close"]) + 1e-9)
    df["signal_strength"] = df["macd_hist"] / (df["atr"] + 1e-9)

    # Safety: replace inf with NaN (later imputer will fix)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ================================
# 2️⃣ Sliding Window Features (optional)
# ================================
def add_price_window_features(df, windows=None):
    if windows is None:
        windows = [3, 5, 10, 20]

    df = df.copy()
    price_cols = ["open", "high", "low", "close", "volume"]
    feature_dict = {}

    def _slope(x):
        if len(x) < 2:
            return np.nan
        try:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        except Exception:
            return np.nan

    for col in price_cols:
        if col not in df.columns:
            continue
        for w in windows:
            roll = df[col].rolling(w)

            feature_dict[f"{col}_mean_{w}"] = roll.mean()
            feature_dict[f"{col}_std_{w}"] = roll.std()
            feature_dict[f"{col}_min_{w}"] = roll.min()
            feature_dict[f"{col}_max_{w}"] = roll.max()
            feature_dict[f"{col}_slope_{w}"] = roll.apply(_slope, raw=True)

    features_df = pd.DataFrame(feature_dict, index=df.index)
    out = pd.concat([df, features_df], axis=1)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


# ================================
# 3️⃣ Full Pipeline
# ================================
def extract_all_features(df, windows=None, include_window_features=True):
    if windows is None:
        windows = [3, 5, 10, 20]

    print("Please wait working with extract_buy_sell_hold_features...")
    df1 = extract_buy_sell_hold_features(df)
    print("#" * 80)

    if include_window_features:
        print("Please wait working with add_price_window_features...")
        df2 = add_price_window_features(df1, windows)
        print("#" * 80)
        return df2

    return df1
