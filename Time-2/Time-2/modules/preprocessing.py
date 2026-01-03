import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def rename_col(df):
    """
    Makes column names consistent:
    open, high, low, close, volume
    Also creates Date column if time exists.
    """
    out = df.copy()

    # Create Date if "time" exists and looks like unix seconds
    if "time" in out.columns and "Date" not in out.columns:
        try:
            out["Date"] = pd.to_datetime(out["time"], unit="s")
        except Exception:
            # if conversion fails, ignore
            pass

    # robust rename: handle both cases
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",

        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
    }

    out.rename(columns=rename_map, inplace=True)

    return out


def handling_nan_after_feature_generate(df_f, target_col=None):
    """
    Cleaner handling:
    - Do NOT drop all rows with NaN (imputer will handle features).
    - Drop only rows where target is NaN (if target_col provided).
    - Drop non-numeric columns except target/entry columns if numeric.
    """
    out = df_f.copy()

    # Drop rows only if target missing
    if target_col is not None and target_col in out.columns:
        out = out.dropna(subset=[target_col]).copy()

    # Drop non-numeric columns (Date etc.)
    non_numeric_cols = out.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        out = out.drop(columns=list(non_numeric_cols), errors="ignore")

    return out


def prepare_dataset_for_model(X_selected, y):
    """
    Returns:
      X_processed: scaled features
      y_mapped: mapped labels for classifiers
               supports BOTH:
                 raw labels   {-1, 0, 1}  -> mapped {1, 0, 2}
                 mapped labels {0, 1, 2}  -> stays as-is
      pipe: fitted preprocessing pipe for use on test data
    """
    X_selected = X_selected.copy()

    # Ensure y is a Series
    y = pd.Series(y).copy()

    # Safety: handle inf/NaN in y (rare but your crash shows it can happen)
    y = y.replace([float("inf"), float("-inf")], pd.NA)

    # Drop rows where y is missing
    valid_mask = y.notna()
    X_selected = X_selected.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    # Convert y to int safely (after cleaning)
    y = y.astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])

    X_processed = pipe.fit_transform(X_selected)

    # ✅ Robust mapping:
    uniq = set(y.unique().tolist())

    # Case A: raw {-1,0,1}
    if uniq.issubset({-1, 0, 1}):
        y_mapped = y.map({-1: 1, 0: 0, 1: 2}).astype(int)

    # Case B: already-mapped {0,1,2}
    elif uniq.issubset({0, 1, 2}):
        y_mapped = y.astype(int)

    else:
        raise ValueError(f"[prepare_dataset_for_model] Unexpected labels: {sorted(list(uniq))}")

    return X_processed, y_mapped, pipe
