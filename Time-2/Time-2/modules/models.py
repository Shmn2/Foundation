import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier


# ============================================================
# Helper: class weights (for imbalance, time-series safe)
# ============================================================
def compute_sample_weights(y):
    """
    y: array-like (mapped classes like {0,1,2})
    returns sample_weight array for each row
    """
    cnt = Counter(y)
    n = len(y)
    k = len(cnt)
    class_weight = {c: n / (k * cnt[c]) for c in cnt}  # balanced formula
    w = np.array([class_weight[yy] for yy in y], dtype=float)
    return w, class_weight


# ============================================================
# XGBoost Train (No ADASYN/SMOTE)
# ============================================================
def xgbmodel(X_processed, y_mapped):
    """
    Train XGBoost with time-aware split, using sample weights for imbalance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_mapped, test_size=0.2, shuffle=False
    )

    sample_w, cw = compute_sample_weights(y_train)
    print("Class weights:", cw)

    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )

    xgb_model.fit(X_train, y_train, sample_weight=sample_w)

    y_pred = xgb_model.predict(X_test)
    print("\n====== 20% Test Classification Report (XGB + sample_weight) ======")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))

    return xgb_model


# ============================================================
# TimeSeries K-Fold evaluation (proper for time-series)
# ============================================================
def xgbmodel_timeseries_kfold(X_processed, y_mapped, n_splits=5):
    """
    TimeSeriesSplit evaluation (no shuffling).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_list, f1_list = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_processed), 1):
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y_mapped.iloc[train_idx], y_mapped.iloc[test_idx]

        sample_w, _ = compute_sample_weights(y_train)

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss"
        )
        model.fit(X_train, y_train, sample_weight=sample_w)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        acc_list.append(acc)
        f1_list.append(f1)

        print(f"\nFold {fold}: Accuracy={acc:.4f}, F1 Macro={f1:.4f}")

    print("\n=========== TimeSeries K-FOLD RESULTS ===========")
    print("Accuracies:", acc_list)
    print("Mean Accuracy:", float(np.mean(acc_list)))
    print("F1 Macros:", f1_list)
    print("Mean F1 Macro:", float(np.mean(f1_list)))

    return float(np.mean(acc_list)), float(np.mean(f1_list))


# ============================================================
# Save / Load model
# ============================================================
def save_model(pipe, features, model):
    models_path = os.path.join(os.getcwd(), "models")
    os.makedirs(models_path, exist_ok=True)

    joblib.dump(pipe, os.path.join(models_path, "preprocessing_pipe.pkl"))
    joblib.dump(features, os.path.join(models_path, "selected_features.pkl"))
    joblib.dump(model, os.path.join(models_path, "xgb_model.pkl"))


def load_model():
    models_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(models_path):
        raise OSError("Model directory missing.")

    pipe = joblib.load(os.path.join(models_path, "preprocessing_pipe.pkl"))
    selected_features = joblib.load(os.path.join(models_path, "selected_features.pkl"))
    model = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
    return pipe, selected_features, model


# ============================================================
# Prediction on new dataset (unknown dataset generalization)
# ============================================================
def mapped_to_signal(pred):
    """
    Inverse mapping of prepare_dataset_for_model():
      {-1:1, 0:0, 1:2}
    so inverse is:
      0 -> 0
      1 -> -1
      2 -> +1
    """
    inv = {0: 0, 1: -1, 2: 1}
    return pd.Series(pred).map(inv).astype(int)


def predict_with_new_dataset(X_new, pipe, model, test_df_features):
    X_new_processed = pipe.transform(X_new)
    y_pred = model.predict(X_new_processed)

    unique, counts = np.unique(y_pred, return_counts=True)
    print("Pred distribution (mapped):", dict(zip(unique, counts)))

    sig = mapped_to_signal(y_pred)

    # align index
    sig.index = test_df_features.index

    out = test_df_features.copy()
    out["Signal"] = sig

    print("\nHead of predicted dataset:")
    print(out.head())
    print("\nSignal counts:", out["Signal"].value_counts().to_dict())

    return out
