import numpy as np
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# --------------------------
# 1) Preprocessing (NaN fix + Scaling)
# --------------------------
def preprocess_features(X):
    """
    Returns:
      Xp (np.array): transformed features
      feature_names (list): original column names
      pipe (Pipeline): fitted preprocessing pipeline
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    Xp = pipe.fit_transform(X)
    feature_names = list(X.columns)
    return Xp, feature_names, pipe


# --------------------------
# 2) L1 selector (Logistic Regression)
# --------------------------
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def l1_selector(X, y, C=0.05):
    model = LogisticRegression(
        penalty="l1",
        solver="saga",     # supports multiclass
        C=C,
        max_iter=5000,
        n_jobs=-1
    )
    selector = SelectFromModel(model)
    selector.fit(X, y)
    return selector.get_support()




# --------------------------
# 3) Tree-based selector (RandomForest importance)
# --------------------------
def rf_selector(Xp, y, k_best=50):
    """
    Uses RandomForest feature importances with SelectFromModel.
    Then optionally caps to top-k by sorting importances.
    """
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xp, y)

    importances = rf.feature_importances_
    if len(importances) == 0:
        return np.array([False] * Xp.shape[1])

    # pick top-k indices
    k = min(k_best, Xp.shape[1])
    top_idx = np.argsort(importances)[::-1][:k]
    mask = np.zeros(Xp.shape[1], dtype=bool)
    mask[top_idx] = True
    return mask


# --------------------------
# 4) Mutual Information selector
# --------------------------
def mi_selector(Xp, y, k_best=50):
    k = min(k_best, Xp.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k)
    sel.fit(Xp, y)
    return sel.get_support()


# --------------------------
# 5) Combined voting
# --------------------------
def combined_feature_selection(Xp, y, feature_names, k_best):
    l1_mask = l1_selector(Xp, y)
    rf_mask = rf_selector(Xp, y, k_best=k_best)
    mi_mask = mi_selector(Xp, y, k_best=k_best)

    votes = l1_mask.astype(int) + rf_mask.astype(int) + mi_mask.astype(int)

    # keep features with >=2 votes
    selected = np.array(feature_names)[votes >= 2]
    masks = {"l1": l1_mask, "rf": rf_mask, "mi": mi_mask}

    return selected, votes, masks


# --------------------------
# 6) Master function
# --------------------------
def select_best_features(X, y, k_best=20):
    """
    X: pandas DataFrame
    y: pandas Series / array-like
    """
    Xp, feature_names, preproc = preprocess_features(X)

    selected, votes, masks = combined_feature_selection(Xp, y, feature_names, k_best)

    # Fallback: if voting selects too few, just take MI top-k
    if len(selected) < min(5, X.shape[1]):
        mi_mask = mi_selector(Xp, y, k_best=k_best)
        selected = np.array(feature_names)[mi_mask]

    return selected, votes, masks, preproc
