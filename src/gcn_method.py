"""
StableOmicsNet — Graph Convolutional Network (the network model).

A 2-layer GCN over a PATIENT-SIMILARITY graph: nodes = patients, node features
= concatenated omics, edges = k-NN similarity. Trained transductively (test
nodes present with features but hidden labels), exactly like MOGONET.

Fits the same interface as the other methods:
    y_pred, importances = gcn(Xtr, ytr, Xte, feat_names, seed=0, top_k=50)

Biomarker importance = input-gradient saliency: |d(predicted logit)/d(feature)|
averaged over the test nodes. Pure PyTorch — no torch-geometric needed.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_adj(X, k=10):
    """Symmetric-normalized adjacency of a cosine k-NN graph with self-loops."""
    A = kneighbors_graph(X, n_neighbors=min(k, len(X) - 1),
                         mode="connectivity", metric="cosine",
                         include_self=False)
    A = A.maximum(A.T).toarray().astype(np.float32)   # symmetrize
    np.fill_diagonal(A, 1.0)                           # self-loops
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)


class _GCN(nn.Module):
    def __init__(self, in_dim, hid, n_cls, p=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid)
        self.lin2 = nn.Linear(hid, n_cls)
        self.p = p

    def forward(self, x, A):
        h = F.relu(A @ self.lin1(x))
        h = F.dropout(h, self.p, self.training)
        return A @ self.lin2(h)


def gcn(Xtr, ytr, Xte, feat_names, seed=0, top_k=50,
        hid=64, epochs=200, k=10, lr=0.01, weight_decay=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_tr = len(ytr)

    sc = StandardScaler().fit(Xtr)                    # fit on train only
    Xs = np.vstack([sc.transform(Xtr), sc.transform(Xte)]).astype(np.float32)
    A = _norm_adj(Xs, k=k)

    le = LabelEncoder().fit(ytr)                       # handles int OR string labels
    y_idx = le.transform(ytr)
    n_cls = len(le.classes_)

    Xt = torch.tensor(Xs, requires_grad=True, device=dev)
    At = torch.tensor(A, device=dev)
    yt = torch.tensor(y_idx, dtype=torch.long, device=dev)
    idx = torch.arange(n_tr, device=dev)

    model = _GCN(Xs.shape[1], hid, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt, At)[idx], yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_idx = model(Xt, At)[n_tr:].argmax(1).cpu().numpy()
    pred = le.inverse_transform(pred_idx)             # back to original labels

    # saliency importance on the test nodes
    if Xt.grad is not None:
        Xt.grad.zero_()
    model.zero_grad()
    logits = model(Xt, At)[n_tr:]
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    sal = Xt.grad[n_tr:].abs().mean(0).cpu().numpy()
    imp = {f: float(s) for f, s in zip(feat_names, sal)}
    return pred, imp
