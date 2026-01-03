from backtesting import Backtest, Strategy
import pandas as pd


class SignalBandStrategy(Strategy):
    def init(self):
        self.last_signal = 0

    def next(self):
        signal = int(self.data.Signal[-1])

        if signal == 0:
            return  # hold

        if signal != self.last_signal:
            # close opposite / previous position
            if self.position:
                self.position.close()

            if signal == 1:
                self.buy()
            elif signal == -1:
                self.sell()

            self.last_signal = signal


def run_backtesting_simulator(df, cash=10000, commission=0.002, plot=False, signal_col="Signal"):
    """
    Robust backtesting runner:
    - supports Signal or EntrySignal
    - supports lowercase open/high/low/close
    - does NOT mutate original df
    """

    bt_df = df.copy()

    # 1) Ensure we have a signal column named "Signal"
    if signal_col not in bt_df.columns:
        raise ValueError(f"Missing required signal column: {signal_col}")

    if signal_col != "Signal":
        bt_df["Signal"] = bt_df[signal_col]
    bt_df["Signal"] = bt_df["Signal"].fillna(0).astype(int)

    # 2) Backtesting.py prefers OHLC named Open/High/Low/Close
    rename_map = {}
    if "open" in bt_df.columns: rename_map["open"] = "Open"
    if "high" in bt_df.columns: rename_map["high"] = "High"
    if "low" in bt_df.columns: rename_map["low"] = "Low"
    if "close" in bt_df.columns: rename_map["close"] = "Close"
    if rename_map:
        bt_df.rename(columns=rename_map, inplace=True)

    # must have Close at minimum
    if "Close" not in bt_df.columns:
        raise ValueError("Backtesting requires a 'Close' column (or 'close' to be renamed).")

    # 3) Set index for plotting/backtesting
    if "Date" in bt_df.columns:
        # ensure datetime
        try:
            bt_df["Date"] = pd.to_datetime(bt_df["Date"])
        except Exception:
            pass
        bt_df = bt_df.set_index("Date")
    else:
        # if Date missing, use integer index
        bt_df = bt_df.reset_index(drop=True)

    bt = Backtest(
    bt_df,
    SignalBandStrategy,
    cash=cash,
    commission=commission,
    exclusive_orders=True,
    finalize_trades=True
)


    stats = bt.run()

    if plot:
        bt.plot(resample=False)

    return stats
