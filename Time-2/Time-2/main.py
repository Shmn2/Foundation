#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

from modules.dataset_loader import load_dataset, load_test_dataset
from modules.preprocessing import (
    rename_col,
    handling_nan_after_feature_generate,
    prepare_dataset_for_model
)
from modules.chart import generate_signal_plot

from modules.signal_label_processing import (
    generate_signal_only_extrema,
    shift_signals,
    enforce_min_distance,
    remove_low_volatility_signals,
    filter_by_slope,
    signal_propagate_for_plot_only
)

from modules.feature_generate import extract_all_features
from modules.feature_selection import select_best_features

# Backtesting
from modules.simulator import run_backtesting_simulator

# Classical ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Deep Learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ============================================================
# Stable mapping (EntrySignal -> mapped classes)
# HOLD always 0 so confidence filter + mapping back is correct.
# ============================================================
def build_stable_mappings():
    """
    EntrySignal raw:  -1 = SELL, 0 = HOLD, +1 = BUY
    Mapped labels:     1 = SELL, 0 = HOLD, 2 = BUY  (HOLD always 0)
    """
    mapping_forward = {-1: 1, 0: 0, 1: 2}
    mapping_back = {0: 0, 1: -1, 2: 1}
    return mapping_forward, mapping_back


# ============================================================
# Helpers
# ============================================================
def compute_sample_weights(y):
    cnt = Counter(y)
    n = len(y)
    k = len(cnt)
    cw = {c: n / (k * cnt[c]) for c in cnt}
    w = np.array([cw[int(v)] for v in y], dtype=float)
    return w, cw


def print_bt_stats(stats, title="Backtest"):
    print(f"\n===== {title} =====")
    keys = [
        "Start", "End", "# Trades", "Win Rate [%]",
        "Return [%]", "Max. Drawdown [%]",
        "Profit Factor", "Sharpe Ratio",
        "Avg. Trade [%]", "Best Trade [%]", "Worst Trade [%]"
    ]
    for k in keys:
        if k in stats:
            print(f"{k:20s}: {stats[k]}")


def backtest_entrysignal(df, plot=False):
    bt_df = df.copy()
    if "EntrySignal" not in bt_df.columns:
        raise ValueError("EntrySignal column missing for backtest.")
    stats = run_backtesting_simulator(bt_df, plot=plot, signal_col="EntrySignal")
    return stats


def choose_best_predictor(classical_results: dict, dl_results: dict) -> str:
    """
    Pick best model primarily by Entry_only_F1_macro, then F1_weighted.
    Returns model name string compatible with predictor argument.
    """
    candidates = {}

    for name, res in (classical_results or {}).items():
        e = res.get("Entry_only_F1_macro")
        fw = res.get("F1_weighted")
        candidates[name] = (float(e) if e is not None else -1.0,
                            float(fw) if fw is not None else -1.0)

    for name, res in (dl_results or {}).items():
        e = res.get("Entry_only_F1_macro")
        fw = res.get("F1_weighted")
        candidates[name] = (float(e) if e is not None else -1.0,
                            float(fw) if fw is not None else -1.0)

    if not candidates:
        return "XGBoost"

    best = sorted(candidates.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)[0][0]
    return best


# ============================================================
# TEST-SIDE FIX: robust datetime parser
# ============================================================
def safe_parse_datetime(s: pd.Series) -> pd.Series:
    """
    Tries multiple datetime parsing strategies:
    1) unix seconds
    2) unix milliseconds
    3) generic string parsing
    Returns pd.Series datetime64 with NaT where failed.
    """
    s = pd.Series(s).copy()

    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")

    if np.issubdtype(s.dtype, np.number):
        dt_s = pd.to_datetime(s, unit="s", errors="coerce")
        if dt_s.notna().mean() > 0.8:
            return dt_s
        dt_ms = pd.to_datetime(s, unit="ms", errors="coerce")
        if dt_ms.notna().mean() > 0.8:
            return dt_ms

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        dt_s = pd.to_datetime(s_num, unit="s", errors="coerce")
        if dt_s.notna().mean() > 0.8:
            return dt_s
        dt_ms = pd.to_datetime(s_num, unit="ms", errors="coerce")
        if dt_ms.notna().mean() > 0.8:
            return dt_ms

    return pd.to_datetime(s.astype(str), errors="coerce", infer_datetime_format=True)


def ensure_datetime_index_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make DateTimeIndex if possible (prevents backtesting warnings/misalignment).
    Tries: Date -> time -> existing index
    """
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        return out

    if "Date" in out.columns:
        dt = pd.to_datetime(out["Date"], errors="coerce")
        if dt.notna().mean() > 0.8:
            out = out.assign(Date=dt).set_index("Date")
        return out

    if "time" in out.columns:
        dt = safe_parse_datetime(out["time"])
        if dt.notna().mean() > 0.8:
            out = out.assign(Date=dt).set_index("Date")
        return out

    return out


def apply_min_trade_distance(df: pd.DataFrame, signal_col="Signal", min_bars=20) -> pd.DataFrame:
    out = df.copy()
    if signal_col not in out.columns:
        return out
    tmp = out.rename(columns={signal_col: "Signal"}).copy()
    tmp2 = enforce_min_distance(tmp, min_bars=min_bars)
    out[signal_col] = tmp2["Signal"].astype(int).values
    return out


# ============================================================
# Deep Learning FIX: build REAL sequences (lookback windows)
# ============================================================
def make_sequences(X_2d: np.ndarray, y: np.ndarray, lookback: int):
    """
    X_2d: (N, F)
    y:    (N,)
    returns:
      X_seq: (N-lookback+1, lookback, F)
      y_seq: (N-lookback+1,)  label aligned at end of window
    """
    X_2d = np.asarray(X_2d)
    y = np.asarray(y)

    N, F = X_2d.shape
    if N <= lookback:
        raise ValueError(f"Not enough rows ({N}) for lookback={lookback}")

    X_seq = np.zeros((N - lookback + 1, lookback, F), dtype=np.float32)
    y_seq = np.zeros((N - lookback + 1,), dtype=np.int64)

    for i in range(lookback, N + 1):
        X_seq[i - lookback] = X_2d[i - lookback:i]
        y_seq[i - lookback] = y[i - 1]  # label at last bar of window

    return X_seq, y_seq


def build_cnn_1d(input_shape, n_classes=3):
    inp = layers.Input(shape=input_shape)  # (lookback, F)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="CNN1D")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm(input_shape, n_classes=3):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(96, return_sequences=False)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru(input_shape, n_classes=3):
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(96, return_sequences=False)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="GRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_dl_models(
    X_train_2d, y_train,
    X_test_2d, y_test,
    class_weight: dict,
    lookback=48,
    epochs=30,
    batch_size=256,
    patience=5,
    out_dir="dl_models"
):
    if not TF_AVAILABLE:
        print("[DL] TensorFlow not installed. Install with: pip install tensorflow")
        return {}

    os.makedirs(out_dir, exist_ok=True)

    X_train_seq, y_train_seq = make_sequences(X_train_2d, y_train, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test_2d, y_test, lookback)

    input_shape = X_train_seq.shape[1:]  # (lookback, F)

    builders = {
        "CNN1D": build_cnn_1d,
        "LSTM": build_lstm,
        "GRU": build_gru
    }

    results = {}

    for name, fn in builders.items():
        print(f"\n[DL] Training {name} ...")
        model = fn(input_shape=input_shape, n_classes=3)

        ckpt_path = os.path.join(out_dir, f"{name}.keras")
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(2, patience // 2), factor=0.5),
        ]

        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            class_weight=class_weight,
            callbacks=cbs
        )

        proba = model.predict(X_test_seq, verbose=0)
        pred = np.argmax(proba, axis=1)

        acc_all = accuracy_score(y_test_seq, pred)
        f1_macro = f1_score(y_test_seq, pred, average="macro")
        f1_weighted = f1_score(y_test_seq, pred, average="weighted")

        mask_entry = (y_test_seq != 0)
        if mask_entry.sum() > 0:
            entry_acc = accuracy_score(y_test_seq[mask_entry], pred[mask_entry])
            entry_f1_macro = f1_score(y_test_seq[mask_entry], pred[mask_entry], average="macro")
        else:
            entry_acc = None
            entry_f1_macro = None

        print(f"[DL:{name}] Accuracy(all)={acc_all:.4f}  F1_macro={f1_macro:.4f}  F1_weighted={f1_weighted:.4f}")
        print(f"[DL:{name}] Entry-only Accuracy={entry_acc}  Entry-only F1_macro={entry_f1_macro}")
        print(classification_report(y_test_seq, pred, zero_division=0))

        results[name] = {
            "Accuracy_all": float(acc_all),
            "F1_macro": float(f1_macro),
            "F1_weighted": float(f1_weighted),
            "Entry_only_accuracy": entry_acc,
            "Entry_only_F1_macro": entry_f1_macro,
            "SavedModel": ckpt_path
        }

    return results


# ============================================================
# Classical ML training
# ============================================================
def train_classical_models(X_train, y_train, X_test, y_test):
    sample_w, cw = compute_sample_weights(y_train)
    print("Class weights used:", cw)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True
        ),
        "XGBoost": XGBClassifier(
            n_estimators=1200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            gamma=0.5,
            eval_metric="mlogloss"
        )
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train, sample_weight=sample_w)
        except TypeError:
            model.fit(X_train, y_train)

        fitted_models[name] = model
        pred = model.predict(X_test)

        acc_all = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")
        f1_weighted = f1_score(y_test, pred, average="weighted")
        prec_macro = precision_score(y_test, pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, pred, average="macro", zero_division=0)

        mask_entry = (y_test != 0)
        if mask_entry.sum() > 0:
            entry_acc = accuracy_score(y_test[mask_entry], pred[mask_entry])
            entry_f1_macro = f1_score(y_test[mask_entry], pred[mask_entry], average="macro")
            entry_prec_macro = precision_score(y_test[mask_entry], pred[mask_entry], average="macro", zero_division=0)
            entry_rec_macro = recall_score(y_test[mask_entry], pred[mask_entry], average="macro", zero_division=0)
        else:
            entry_acc = entry_f1_macro = entry_prec_macro = entry_rec_macro = np.nan

        results[name] = {
            "Accuracy_all": float(acc_all),
            "F1_macro": float(f1_macro),
            "F1_weighted": float(f1_weighted),
            "Precision_macro": float(prec_macro),
            "Recall_macro": float(rec_macro),
            "Entry_only_accuracy": float(entry_acc) if not np.isnan(entry_acc) else None,
            "Entry_only_F1_macro": float(entry_f1_macro) if not np.isnan(entry_f1_macro) else None,
            "Entry_only_Precision_macro": float(entry_prec_macro) if not np.isnan(entry_prec_macro) else None,
            "Entry_only_Recall_macro": float(entry_rec_macro) if not np.isnan(entry_rec_macro) else None,
        }

        print(f"\n==== {name} Classification Report (ALL) ====")
        print(classification_report(y_test, pred, zero_division=0))
        print(f"Accuracy (all bars): {acc_all:.4f}")
        print(f"Macro F1 (all): {f1_macro:.4f} | Weighted F1 (all): {f1_weighted:.4f}")

        if not np.isnan(entry_acc):
            print("\n---- Entry-only metrics (BUY/SELL only, HOLD excluded) ----")
            print(f"Entry-only accuracy: {entry_acc:.4f}")
            print(f"Entry-only Macro F1: {entry_f1_macro:.4f}")
            print(f"Entry-only Precision (macro): {entry_prec_macro:.4f}")
            print(f"Entry-only Recall (macro): {entry_rec_macro:.4f}")
        else:
            print("Entry-only metrics: N/A (no entry labels in test split)")

        uniq, cnts = np.unique(pred, return_counts=True)
        print("Pred distribution:", dict(zip(uniq.tolist(), cnts.tolist())))

    return results, fitted_models


# ============================================================
# Visualization (plot only)
# ============================================================
def visualize_dataset(df, processed, limit=3000):
    tmp = df.reset_index(drop=True).copy()
    generate_signal_plot(tmp, val_limit=limit)

    ex = generate_signal_only_extrema(tmp, cluster_length=30, use_hilo=True)
    generate_signal_plot(ex, val_limit=limit)

    sh = shift_signals(ex, delay=2)
    generate_signal_plot(sh, val_limit=limit)

    prop = signal_propagate_for_plot_only(sh)
    generate_signal_plot(prop, val_limit=limit)

    if "EntrySignal" in processed.columns:
        plot_df = processed.rename(columns={"EntrySignal": "Signal"}).copy()
    else:
        plot_df = processed.copy()

    generate_signal_plot(plot_df, val_limit=limit)


# ============================================================
# Pipeline
# ============================================================
class SignalMLPipeline:
    def __init__(
        self,
        data_dir_,
        file_name_,
        test_file_path_,
        n_features=20,
        pred_conf_threshold=None,
        hold_ratio=3,
        pred_min_bars=20,
        pred_atr_filter=False,
        pred_atr_percentile=20,
        pred_atr_period=14,
        dl_epochs=30,
        dl_batch=256,
        dl_patience=5,
        dl_lookback=48,
        predictor="auto",             # NEW
    ):
        self.data_dir = data_dir_
        self.file_name = file_name_
        self.test_file_path = test_file_path_
        self.n_features = n_features
        self.pred_conf_threshold = pred_conf_threshold

        self.hold_ratio = int(hold_ratio)
        self.pred_min_bars = int(pred_min_bars)

        self.pred_atr_filter = bool(pred_atr_filter)
        self.pred_atr_percentile = int(pred_atr_percentile)
        self.pred_atr_period = int(pred_atr_period)

        self.dl_epochs = int(dl_epochs)
        self.dl_batch = int(dl_batch)
        self.dl_patience = int(dl_patience)
        self.dl_lookback = int(dl_lookback)

        self.predictor = str(predictor)

        self.mapping_forward, self.mapping_back = build_stable_mappings()

        self.raw_data = None
        self.dataset = None
        self.df_features = None
        self.selected_features = None

        self.train_pipe = None
        self.classical_models = None

        # NEW: keep results + DL models + best name
        self.classical_results = {}
        self.dl_results = {}
        self.dl_models = {}
        self.best_model_name = None

        self.step_functions = {
            1: ("load", self.load_and_prepare_raw_data),
            2: ("label", self.generate_labels),
            3: ("visualize", self.visualize_current_dataset),
            4: ("features", self.extract_features),
            5: ("select", self._feature_selection_wrapper),
            6: ("train", self._train_wrapper),
            7: ("test", self.test_new_dataset),
        }

    def run_pipeline(self, start_step_=1, end_step_=6):
        print(f"\n>>> Running pipeline from step {start_step_} to {end_step_}\n")
        for step in range(start_step_, end_step_ + 1):
            name, fn = self.step_functions[step]
            print(f"\n=== Step {step}: {name.upper()} ===")
            fn()

    def load_and_prepare_raw_data(self):
        print("Loading dataset...")
        dt = load_dataset(self.data_dir, self.file_name)
        dt = rename_col(dt)

        for c in ["open", "high", "low", "close"]:
            if c not in dt.columns:
                raise ValueError(f"Missing required column '{c}' after rename_col().")

        self.raw_data = dt.reset_index(drop=True)
        print("Loaded rows:", len(self.raw_data))

    def generate_labels(self):
        print("Generating Group-1 EntrySignal + ML Label (EntrySignal)...")

        base = generate_signal_only_extrema(self.raw_data, cluster_length=30, use_hilo=True)
        tuned = shift_signals(base, delay=2)
        tuned = enforce_min_distance(tuned, min_bars=20)
        tuned = remove_low_volatility_signals(tuned, threshold_percentile=20, atr_period=14)
        tuned = filter_by_slope(tuned, look_ahead=24, slope_threshold=0.0)

        tuned = tuned.rename(columns={"Signal": "EntrySignal"}).copy()
        tuned["Label"] = tuned["EntrySignal"].astype(int)
        print("[Label] Using EntrySignal as ML Label (Group-1 aligned).")

        self.dataset = tuned.reset_index(drop=True)

        print("\nBacktesting (EntrySignal)...")
        stats = backtest_entrysignal(self.dataset, plot=False)
        print_bt_stats(stats, title="EntrySignal Backtest")

        os.makedirs(self.data_dir, exist_ok=True)
        save_path = os.path.join(self.data_dir, "group1_entrysignal_label.csv")
        self.dataset.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    def visualize_current_dataset(self):
        if self.raw_data is None:
            self.load_and_prepare_raw_data()
        if self.dataset is None:
            self.generate_labels()
        visualize_dataset(self.raw_data, self.dataset, limit=3000)

    def extract_features(self):
        print("Extracting features...")
        df_feat = extract_all_features(self.dataset)
        df_feat = handling_nan_after_feature_generate(df_feat, target_col="Label")
        self.df_features = df_feat
        print("Features shape:", self.df_features.shape)

    def feature_selection(self):
        print("Performing feature selection...")

        df = self.df_features.dropna(subset=["Label"]).copy()
        y_raw = df["Label"].astype(int).values
        y = pd.Series(y_raw).map(self.mapping_forward).astype(int).values  # 0/1/2 only

        X = df.drop(columns=["EntrySignal", "Label"], errors="ignore")
        X = X.drop(columns=["time", "Date"], errors="ignore")
        X = X.fillna(X.mean())

        selected_features, votes, masks, _ = select_best_features(X, y, self.n_features)
        self.selected_features = [f for f in list(selected_features) if f not in ["time", "Date"]]
        pd.Series(self.selected_features).to_csv("selected_features.csv", index=False)

        print("Selected Features:", self.selected_features)
        return X, y

    def _feature_selection_wrapper(self):
        self.X, self.y = self.feature_selection()

    def train_model(self, X, y):
        print("Preparing dataset for model training...")
        x_selected = X[self.selected_features]

        x_processed, y_mapped, pipe = prepare_dataset_for_model(x_selected, y)
        self.train_pipe = pipe

        # Downsample HOLD to reduce imbalance
        y_series = pd.Series(y_mapped)
        hold_idx = y_series[y_series == 0].index
        trade_idx = y_series[y_series != 0].index

        if len(trade_idx) > 0 and self.hold_ratio is not None:
            max_hold = min(len(hold_idx), int(self.hold_ratio) * len(trade_idx))
            rng = np.random.RandomState(42)
            keep_hold = rng.choice(hold_idx, size=max_hold, replace=False) if max_hold > 0 else np.array([], dtype=int)

            keep_idx = np.concatenate([trade_idx.values, keep_hold])
            keep_idx.sort()

            x_processed = x_processed[keep_idx]
            y_mapped = y_series.iloc[keep_idx].values

            print(f"\n[Downsample] trades={len(trade_idx)}, holds_kept={len(keep_hold)} (ratio={self.hold_ratio}x)")
        else:
            print("\n[Downsample] skipped (no trades found or hold_ratio not set).")

        print("\nLabel distribution (mapped, after downsample):")
        print(pd.Series(y_mapped).value_counts())
        print(pd.Series(y_mapped).value_counts(normalize=True))

        X_train, X_test, y_train, y_test = train_test_split(
            x_processed, y_mapped, test_size=0.2, shuffle=False
        )

        print("\n=== Classical ML Models (weighted + macro + entry-only metrics) ===")
        classical_results, classical_models = train_classical_models(X_train, y_train, X_test, y_test)
        self.classical_results = classical_results
        self.classical_models = classical_models

        print("\n=== Deep Learning Models (CNN1D / LSTM / GRU) ===")
        _, cw = compute_sample_weights(y_train)
        cw = {int(k): float(v) for k, v in cw.items()}

        dl_results = train_dl_models(
            X_train, y_train,
            X_test, y_test,
            class_weight=cw,
            lookback=self.dl_lookback,
            epochs=self.dl_epochs,
            batch_size=self.dl_batch,
            patience=self.dl_patience,
            out_dir=os.path.join(self.data_dir, "dl_models")
        )
        self.dl_results = dl_results

        # Load saved DL models (for Step-7)
        self.dl_models = {}
        if TF_AVAILABLE:
            for nm in ["CNN1D", "LSTM", "GRU"]:
                p = os.path.join(self.data_dir, "dl_models", f"{nm}.keras")
                if os.path.exists(p):
                    self.dl_models[nm] = tf.keras.models.load_model(p)

        # Decide best predictor automatically
        self.best_model_name = choose_best_predictor(self.classical_results, self.dl_results)
        print(f"\n[BEST] Auto-selected best predictor = {self.best_model_name}")

    def _train_wrapper(self):
        if not hasattr(self, "X") or not hasattr(self, "y"):
            raise RuntimeError("Run feature selection before training.")
        self.train_model(self.X, self.y)

    def test_new_dataset(self):
        print("Loading external test dataset...")
        test_df = load_test_dataset(self.test_file_path)
        test_df = rename_col(test_df).reset_index(drop=True)

        # Build Date column robustly for backtest
        dt = None
        if "Date" in test_df.columns:
            dt = pd.to_datetime(test_df["Date"], errors="coerce")
        elif "time" in test_df.columns:
            dt = safe_parse_datetime(test_df["time"])

        if dt is not None and dt.notna().mean() > 0.8:
            test_df["Date"] = dt

        print("Extracting features from test dataset...")
        test_feat = extract_all_features(test_df)
        test_feat = handling_nan_after_feature_generate(test_feat)

        missing = [c for c in self.selected_features if c not in test_feat.columns]
        if missing:
            raise ValueError(f"Missing selected features in test dataset: {missing[:10]} ...")

        X_new = test_feat[self.selected_features].copy()
        X_new = X_new.fillna(X_new.mean())

        if self.train_pipe is None:
            raise RuntimeError("train_pipe missing. Train first.")
        X_new_processed = self.train_pipe.transform(X_new)

        # Decide which model to use in Step-7
        predictor = self.predictor
        if predictor.lower() == "auto":
            predictor = self.best_model_name or "GRU"

        print(f"\n[TEST] Using predictor = {predictor}")

        pred_mapped = None
        p_trade_full = None  # for optional confidence filtering

        # ---------- DL prediction ----------
        if predictor in ["GRU", "LSTM", "CNN1D"]:
            if not TF_AVAILABLE:
                raise RuntimeError("TensorFlow not available but DL predictor requested.")
            if predictor not in self.dl_models:
                raise RuntimeError(f"DL model {predictor} not loaded/saved. Train first.")
            model = self.dl_models[predictor]

            X_seq, _ = make_sequences(X_new_processed, np.zeros(len(X_new_processed)), self.dl_lookback)
            proba = model.predict(X_seq, verbose=0)  # (N-lookback+1, 3)
            pred_seq = np.argmax(proba, axis=1)

            # Align: first lookback-1 rows -> HOLD
            pred_mapped = np.zeros(len(X_new_processed), dtype=int)
            pred_mapped[self.dl_lookback - 1:] = pred_seq

            # Confidence: p_trade = 1 - p_hold
            p_trade = 1.0 - proba[:, 0]
            p_trade_full = np.zeros(len(X_new_processed), dtype=float)
            p_trade_full[self.dl_lookback - 1:] = p_trade

            if self.pred_conf_threshold is not None:
                thr = float(self.pred_conf_threshold)
                pred_mapped = np.where(p_trade_full >= thr, pred_mapped, 0)
                test_feat["P_trade"] = p_trade_full

        # ---------- Classical prediction ----------
        else:
            if self.classical_models is None or predictor not in self.classical_models:
                raise RuntimeError(f"Classical model {predictor} missing. Train first.")
            model = self.classical_models[predictor]
            pred_mapped = model.predict(X_new_processed)

            if self.pred_conf_threshold is not None:
                try:
                    proba = model.predict_proba(X_new_processed)
                    hold_class = 0
                    hold_idx = int(np.where(model.classes_ == hold_class)[0][0])
                    p_trade_full = 1.0 - proba[:, hold_idx]

                    thr = float(self.pred_conf_threshold)
                    pred_mapped = np.where(p_trade_full >= thr, pred_mapped, 0)

                    test_feat["P_trade"] = p_trade_full
                except Exception as e:
                    print("[WARN] predict_proba failed; running without confidence filter.", e)

        pred_signal = pd.Series(pred_mapped).map(self.mapping_back).astype(int)

        out = test_feat.copy()
        out["Signal"] = pred_signal.values

        # Re-attach Date for backtest
        if "Date" in test_df.columns:
            dt2 = pd.to_datetime(test_df["Date"], errors="coerce")
            if dt2.notna().mean() > 0.8 and len(dt2) == len(out):
                out["Date"] = dt2.values

        # OPTIONAL: ATR low-volatility filter
        if self.pred_atr_filter:
            try:
                base_cols = []
                for c in ["open", "high", "low", "close", "volume", "time", "Date"]:
                    if c in test_df.columns:
                        base_cols.append(c)

                tmp_bt = test_df[base_cols].copy()
                tmp_bt["Signal"] = out["Signal"].astype(int).values

                tmp_bt = remove_low_volatility_signals(
                    tmp_bt,
                    threshold_percentile=self.pred_atr_percentile,
                    atr_period=self.pred_atr_period
                )

                out["Signal"] = tmp_bt["Signal"].astype(int).values
            except Exception as e:
                print("[WARN] pred_atr_filter failed; continuing without it.", e)

        # Debounce predicted signals
        out = apply_min_trade_distance(out, signal_col="Signal", min_bars=self.pred_min_bars)

        print("\nUnknown dataset prediction distribution:")
        print(out["Signal"].value_counts())

        if "close" in out.columns:
            generate_signal_plot(out[["close", "Signal"]].copy())

        out_bt = ensure_datetime_index_for_backtest(out)

        try:
            stats = run_backtesting_simulator(out_bt, plot=False, signal_col="Signal")
            print_bt_stats(stats, title=f"Predicted Signal Backtest (Unknown Dataset) [{predictor}]")
        except Exception as e:
            print("Backtest on unknown dataset failed:", e)

        return out


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group-1 Signal ML Pipeline (EntrySignal as Label)")
    parser.add_argument("--start", type=str, default="1", help="Start step (number or name)")
    parser.add_argument("--end", type=str, default="6", help="End step (number or name)")

    parser.add_argument("--predictor", type=str, default="auto",
                        help="auto | GRU | LSTM | CNN1D | XGBoost | RandomForest | LogisticRegression | SVM")

    parser.add_argument("--conf", type=float, default=None,
                        help="If set, only trade when P(trade)=1-P(hold) >= conf. Example: 0.80")

    parser.add_argument("--hold_ratio", type=int, default=3,
                        help="Keep at most hold_ratio * trades HOLD samples in training. Example: 1/2/3")

    parser.add_argument("--pred_min_bars", type=int, default=20,
                        help="Minimum bars between predicted non-zero signals in TEST. Example: 20/40/60")

    parser.add_argument("--pred_atr_filter", action="store_true",
                        help="If set, applies ATR low-volatility filter to predicted Signal in TEST.")
    parser.add_argument("--pred_atr_percentile", type=int, default=20,
                        help="ATR percentile for low-volatility filter in TEST (default 20).")
    parser.add_argument("--pred_atr_period", type=int, default=14,
                        help="ATR period for low-volatility filter in TEST (default 14).")

    # Deep learning knobs
    parser.add_argument("--dl_epochs", type=int, default=30, help="DL epochs (default 30)")
    parser.add_argument("--dl_batch", type=int, default=256, help="DL batch size (default 256)")
    parser.add_argument("--dl_patience", type=int, default=5, help="DL early stopping patience (default 5)")
    parser.add_argument("--dl_lookback", type=int, default=48, help="DL lookback window (candles)")

    args = parser.parse_args()

    step_map = {
        "load": 1,
        "label": 2,
        "visualize": 3,
        "features": 4,
        "select": 5,
        "train": 6,
        "test": 7
    }

    def convert_step(x):
        if x.isdigit():
            return int(x)
        return step_map[x.lower()]

    start_step = convert_step(args.start)
    end_step = convert_step(args.end)

    data_dir = r"C:\Users\Asus\Downloads\Useful\books\Course materials\MScCS\Foundation\FT\Time-2\Time-2\datasets"
    file_name = "Cleaned_Signal_EURUSD_for_training_635_635_60000.csv"
    test_file_path = r"C:\Users\Asus\Downloads\Useful\books\Course materials\MScCS\Foundation\FT\Time-2\Time-2\datasets\GBPUSD_H1_20140525_20251021.csv"

    pipeline = SignalMLPipeline(
        data_dir_=data_dir,
        file_name_=file_name,
        test_file_path_=test_file_path,
        n_features=20,
        pred_conf_threshold=args.conf,
        hold_ratio=args.hold_ratio,
        pred_min_bars=args.pred_min_bars,
        pred_atr_filter=args.pred_atr_filter,
        pred_atr_percentile=args.pred_atr_percentile,
        pred_atr_period=args.pred_atr_period,
        dl_epochs=args.dl_epochs,
        dl_batch=args.dl_batch,
        dl_patience=args.dl_patience,
        dl_lookback=args.dl_lookback,
        predictor=args.predictor
    )

    pipeline.run_pipeline(start_step, end_step)
