"""
StableOmicsNet — integration/selection methods behind ONE interface.

Every method takes early-fused (concatenated) omics and returns:
  - y_pred        : predicted labels for the test set
  - importances   : dict {feature_name -> importance score} over ALL features

This uniform interface lets the stability engine compare very different methods
(trees, sparse linear, univariate) on equal footing. SNF and a GCN (true
network models) plug in here next.

    from methods import METHODS
    y_pred, imp = METHODS["RandomForest"](Xtr, ytr, Xte, feat_names, seed=0)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif


def _imp_dict(feat_names, scores):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    return {f: float(s) for f, s in zip(feat_names, scores)}


# ---- tree ensembles ---------------------------------------------------------
def random_forest(Xtr, ytr, Xte, feat_names, seed=0, top_k=50):
    m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte), _imp_dict(feat_names, m.feature_importances_)


def extra_trees(Xtr, ytr, Xte, feat_names, seed=0, top_k=50):
    m = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte), _imp_dict(feat_names, m.feature_importances_)


# ---- sparse linear (L1) -----------------------------------------------------
def l1_logistic(Xtr, ytr, Xte, feat_names, seed=0, top_k=50):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    m = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                           max_iter=3000, random_state=seed)
    m.fit(Xtr_s, ytr)
    coef = np.abs(m.coef_)                    # (n_classes|1, n_features)
    score = coef.mean(axis=0)
    return m.predict(Xte_s), _imp_dict(feat_names, score)


# ---- univariate ANOVA (+ logistic head for prediction) ----------------------
def univariate_anova(Xtr, ytr, Xte, feat_names, seed=0, top_k=50):
    F, _ = f_classif(Xtr, ytr)
    F = np.nan_to_num(F, nan=0.0)
    top_idx = np.argsort(F)[::-1][:top_k]
    sc = StandardScaler().fit(Xtr[:, top_idx])
    head = LogisticRegression(max_iter=2000, random_state=seed)
    head.fit(sc.transform(Xtr[:, top_idx]), ytr)
    y_pred = head.predict(sc.transform(Xte[:, top_idx]))
    return y_pred, _imp_dict(feat_names, F)


METHODS = {
    "RandomForest": random_forest,
    "ExtraTrees": extra_trees,
    "L1-Logistic": l1_logistic,
    "Univariate-ANOVA": univariate_anova,
}

# Optional network model (needs PyTorch). Registered only if torch is present,
# so the rest of the framework runs without it.
try:
    from gcn_method import gcn as _gcn
    METHODS["GCN"] = _gcn
except Exception as _e:               # torch not installed / import failed
    print(f"[methods] GCN not registered ({type(_e).__name__}). "
          f"Install torch to enable it.")

# Optional second network model: Similarity Network Fusion (needs snfpy).
try:
    from snf_method import snf_method as _snf
    METHODS["SNF"] = _snf
except Exception as _e:
    print(f"[methods] SNF not registered ({type(_e).__name__}). "
          f"Install snfpy to enable it.")
