import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier


def compute_sample_weights(y):
    """
    y: array-like (classes like 0/1/2)
    Returns:
      sample_weight array
      class_weight dict (debug)
    """
    cnt = Counter(y)
    n = len(y)
    k = len(cnt)
    class_weight = {c: n / (k * cnt[c]) for c in cnt}
    w = np.array([class_weight[v] for v in y], dtype=float)
    return w, class_weight


# ---------------------------------------------------------
# TimeSeries K-Fold Evaluation (NO OVERSAMPLING)
# ---------------------------------------------------------
def timeseries_kfold_evaluate(X, y, n_splits=5, name="XGB"):
    """
    X: numpy array
    y: pandas Series or array-like
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_sr = pd.Series(y).reset_index(drop=True)

    acc_list = []
    f1_list = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_df), 1):
        X_tr, X_val = X_df.iloc[train_idx].values, X_df.iloc[val_idx].values
        y_tr, y_val = y_sr.iloc[train_idx].values, y_sr.iloc[val_idx].values

        sample_w, cw = compute_sample_weights(y_tr)

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss"
        )

        model.fit(X_tr, y_tr, sample_weight=sample_w)
        pred = model.predict(X_val)

        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")

        acc_list.append(acc)
        f1_list.append(f1)

        print(f"\n========== {name} Fold {fold} ==========")
        print("Class weights:", cw)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1:.4f}")

    print(f"\n========== {name} TimeSeries K-FOLD SUMMARY ==========")
    print("Accuracies:", acc_list)
    print("Mean Accuracy:", float(np.mean(acc_list)))
    print("F1 Macros:", f1_list)
    print("Mean F1 Macro:", float(np.mean(f1_list)))

    return float(np.mean(acc_list)), float(np.mean(f1_list))


# ---------------------------------------------------------
# Simple 80/20 Time-Aware Evaluation (NO OVERSAMPLING)
# ---------------------------------------------------------
def evaluate_timeseries_holdout(X_train, y_train, X_test, y_test, name="XGB"):
    """
    Train on train, test on last 20% (no shuffle) using sample_weight.
    """
    y_train = pd.Series(y_train).values
    y_test = pd.Series(y_test).values

    sample_w, cw = compute_sample_weights(y_train)

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

    print(f"\n========== {name} (20% Test) ==========")
    print("Class weights:", cw)
    print(classification_report(y_test, pred))
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Macro: {f1:.4f}")

    return acc, f1, model
