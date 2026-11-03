"""
StableOmicsNet — biomarker-stability metrics (the core novelty).

Given several biomarker selections (e.g. top-k features from different random
seeds / bootstraps / methods), quantify how consistent they are.

    from stability import mean_pairwise_jaccard, kuncheva_index, rank_stability

    selections = [set_a, set_b, set_c]      # each a set/list of feature names
    mean_pairwise_jaccard(selections)       # 0..1, higher = more stable
    kuncheva_index(selections, n_total=2000)# -1..1, corrects for chance

    # rank-level stability from importance vectors (dict feat->score):
    rank_stability([imp1, imp2, imp3], top_k=50)
"""
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr, rankdata


def top_k_features(imp: dict, k: int) -> list:
    """
    The k highest-scoring features of one fit, restricted to features with a
    positive score. A sparse method (e.g. L1) that assigns non-zero weight to
    fewer than k features selects only those; zero-score features are NOT used
    as filler, because ordering ties by input position would make the "filler"
    identical across fits and inflate stability.
    """
    ordered = sorted((f for f in imp if imp[f] > 0), key=imp.get, reverse=True)
    return ordered[:k]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(selections) -> float:
    """Average Jaccard over all pairs of selections."""
    sels = [set(s) for s in selections]
    pairs = list(combinations(range(len(sels)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([jaccard(sels[i], sels[j]) for i, j in pairs]))


def kuncheva_index(selections, n_total: int) -> float:
    """
    Kuncheva consistency index, corrected for chance.

    For two subsets A, B of size k drawn from n_total features with
    r = |A n B|:   KI = (r - k^2/n) / (k - k^2/n)
    Subsets of unequal size (a sparse method selecting fewer than k features)
    use the natural extension KI = (r - ka*kb/n) / (min(ka, kb) - ka*kb/n),
    which reduces to the formula above when ka = kb. Returns the mean over
    all pairs.
    """
    sels = [set(s) for s in selections]
    n = n_total
    pairs = list(combinations(range(len(sels)), 2))
    if not pairs:
        return 1.0
    vals = []
    for i, j in pairs:
        ka, kb = len(sels[i]), len(sels[j])
        if min(ka, kb) == 0 or max(ka, kb) >= n:
            raise ValueError("Subset sizes must satisfy 0 < k < n_total")
        r = len(sels[i] & sels[j])
        expected = ka * kb / n
        vals.append((r - expected) / (min(ka, kb) - expected))
    return float(np.mean(vals))


def rank_stability(importances, top_k: int | None = None) -> float:
    """
    Mean pairwise Spearman correlation of feature-importance rankings.

    `importances` is a list of dicts {feature: score}. Features are aligned on
    their union; missing features get score 0. If top_k is given, correlation is
    computed on the union of each pair's top_k features (more sensitive to the
    part of the ranking we actually act on).
    """
    dicts = [dict(d) for d in importances]
    pairs = list(combinations(range(len(dicts)), 2))
    if not pairs:
        return 1.0
    vals = []
    for i, j in pairs:
        di, dj = dicts[i], dicts[j]
        if top_k:
            top_i = sorted(di, key=di.get, reverse=True)[:top_k]
            top_j = sorted(dj, key=dj.get, reverse=True)[:top_k]
            feats = list(set(top_i) | set(top_j))
        else:
            feats = list(set(di) | set(dj))
        vi = np.array([di.get(f, 0.0) for f in feats])
        vj = np.array([dj.get(f, 0.0) for f in feats])
        rho, _ = spearmanr(vi, vj)
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def consensus_ranking(importances, top_k: int | None = None) -> list:
    """
    Aggregate several importance dicts into one consensus feature ranking by
    mean rank (Borda-style). Returns features ordered most->least important.
    This is the 'stable core' selector used later in the pipeline.
    """
    dicts = [dict(d) for d in importances]
    return weighted_consensus_ranking(dicts, np.ones(len(dicts)), top_k)


def _rank_matrix(dicts, feats):
    """
    Rank of every feature in every importance dict (1 = best). Tied scores,
    in particular the many exact zeros of sparse methods, share their average
    rank, so ties are never resolved by input position. Features missing from
    a dict rank below all of its features.
    """
    R = np.empty((len(dicts), len(feats)))
    for i, d in enumerate(dicts):
        s = np.array([d.get(f, -np.inf) for f in feats], dtype=float)
        R[i] = rankdata(-s, method="average")
    return R


def weighted_consensus_ranking(importances, weights, top_k: int | None = None) -> list:
    """
    Consensus ranking in which each importance dict contributes in proportion to
    `weights` (e.g. the stability of the method that produced it).

    Rank aggregation is weighted-mean-rank: features ranked highly by STABLE
    contributors dominate. Weights are clipped at 0 and normalised internally;
    if every weight is 0 the function falls back to the unweighted consensus.
    """
    dicts = [dict(d) for d in importances]
    w = np.asarray(weights, dtype=float)
    if len(w) != len(dicts):
        raise ValueError("weights and importances must have the same length")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return consensus_ranking(dicts, top_k)
    w = w / w.sum()

    feats = sorted(set().union(*[set(d) for d in dicts])) if dicts else []
    if not feats:
        return []
    scores = w @ _rank_matrix(dicts, feats)
    ordered = [feats[i] for i in np.argsort(scores, kind="stable")]
    return ordered[:top_k] if top_k else ordered


if __name__ == "__main__":
    # tiny self-test
    a = ["g1", "g2", "g3", "g4"]
    b = ["g1", "g2", "g3", "g9"]
    c = ["g1", "g2", "g7", "g8"]
    print("jaccard(a,b)          :", round(jaccard(a, b), 3))
    print("mean_pairwise_jaccard :", round(mean_pairwise_jaccard([a, b, c]), 3))
    print("kuncheva (n=100)      :", round(kuncheva_index([a, b, c], 100), 3))
    imp = [{"g1": 5, "g2": 4, "g3": 3}, {"g1": 4, "g2": 5, "g3": 2}]
    print("rank_stability        :", round(rank_stability(imp), 3))
    print("consensus_ranking     :", consensus_ranking(imp))
