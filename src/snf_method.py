"""
StableOmicsNet — Similarity Network Fusion (a second, canonical network model).

SNF (Wang et al., Nat Methods 2014) builds a patient-similarity network per omic
and fuses them into ONE multiplex network by cross-diffusion. It is the reference
multi-omics *network* method, so it complements the GCN.

Self-contained NumPy implementation (no snfpy dependency, so it is robust to
sklearn version changes). Fitted to the shared interface:
  y_pred      : label propagation on the fused network (train-neighbour vote)
  importances : each feature's |correlation| with the top spectral-embedding
                dimensions of the fused graph -> a genuinely NETWORK-based
                biomarker score, distinct from the class-based scores of the
                other methods.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def _omic_blocks(feat_names):
    groups = {}
    for i, f in enumerate(feat_names):
        key = f.split(":", 1)[0] if ":" in f else "all"
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def _scaled_affinity(X, K, mu):
    """Scaled-exponential affinity with local (k-NN) bandwidth (Wang et al.)."""
    sq = np.sum(X * X, 1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    Ds = np.sort(D, 1)
    sigma = Ds[:, 1:K + 1].mean(1) + 1e-10           # mean dist to K neighbours
    eps = (sigma[:, None] + sigma[None, :] + D) / 3 + 1e-10
    return np.exp(-(D ** 2) / (mu * eps))


def _dominate(W, K):
    """Sparse k-NN transition matrix: keep top-K per row, renormalise."""
    n = len(W)
    S = np.zeros_like(W)
    idx = np.argsort(-W, 1)[:, :K]
    rows = np.repeat(np.arange(n), K)
    S[rows, idx.ravel()] = W[rows, idx.ravel()]
    return S / (S.sum(1, keepdims=True) + 1e-10)


def _snf_fuse(affs, K, t, alpha=1.0):
    aff = []
    for W in affs:
        Wn = W / (W.sum(1, keepdims=True) + 1e-10)
        aff.append((Wn + Wn.T) / 2)
    if len(aff) == 1:
        return aff[0]
    S = [_dominate(W, K) for W in affs]
    for _ in range(t):
        Wsum = sum(aff)
        aff = [((S[i] @ (Wsum - aff[i]) @ S[i].T) / (len(aff) - 1)
                + alpha * np.eye(len(aff[i]))) for i in range(len(aff))]
        aff = [(a + a.T) / 2 for a in aff]
    W = sum(aff) / len(aff)
    W = W / (W.sum(1, keepdims=True) + 1e-10)
    return (W + W.T) / 2


def snf_method(Xtr, ytr, Xte, feat_names, seed=0, top_k=50,
               K=20, mu=0.5, t=20, n_emb=10):
    feat_names = np.asarray(feat_names)
    n_tr = len(ytr)
    sc = StandardScaler().fit(Xtr)
    Xs = np.vstack([sc.transform(Xtr), sc.transform(Xte)]).astype(float)
    k = min(K, len(Xs) - 1)

    views = [Xs[:, idx] for idx in _omic_blocks(feat_names)]
    affs = [_scaled_affinity(v, k, mu) for v in views]
    W = _snf_fuse(affs, k, t)

    # prediction: weighted vote from train neighbours in the fused network
    classes = np.unique(ytr)
    Wtt = W[n_tr:, :n_tr]
    scores = np.column_stack([Wtt[:, ytr == c].sum(1) for c in classes])
    pred = classes[scores.argmax(1)]

    # importance: |corr| of each feature with the fused graph's spectral embedding
    d = W.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    L = np.eye(len(W)) - (W * dinv[:, None] * dinv[None, :])
    _, vecs = np.linalg.eigh(L)
    emb = vecs[:, 1:1 + n_emb]
    Xz = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    ez = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    corr = np.abs(Xz.T @ ez) / len(Xs)
    imp = {f: float(s) for f, s in zip(feat_names, corr.max(1))}
    return pred, imp
