"""A minimal histogram gradient booster, so the project needs only numpy.

No network access in this environment, so scikit-learn / LightGBM are not
installable. This also keeps the dependency list to the three libraries the
provided scorer already requires.
"""
from __future__ import annotations

import numpy as np


class HistGBM:
    def __init__(self, n_estimators=200, learning_rate=0.06, max_depth=5,
                 n_bins=64, min_samples=40):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.n_bins, self.min_samples = max_depth, n_bins, min_samples

    def _bin(self, X, fit=False):
        if fit:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return np.column_stack([np.searchsorted(self.edges_[j], X[:, j])
                                for j in range(X.shape[1])]).astype(np.int32)

    def _split(self, B, g, idx):
        best, tot_s, tot_c = None, g[idx].sum(), len(idx)
        for j in range(B.shape[1]):
            nb = len(self.edges_[j]) + 1
            s = np.bincount(B[idx, j], weights=g[idx], minlength=nb)
            c = np.bincount(B[idx, j], minlength=nb)
            cs, cc = np.cumsum(s)[:-1], np.cumsum(c)[:-1]
            ok = (cc >= self.min_samples) & (tot_c - cc >= self.min_samples)
            if not ok.any():
                continue
            gain = np.where(ok, cs ** 2 / np.maximum(cc, 1)
                            + (tot_s - cs) ** 2 / np.maximum(tot_c - cc, 1), -np.inf)
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (gain[k], j, k)
        return best

    def _grow(self, B, g, idx, depth):
        if depth == 0 or len(idx) < 2 * self.min_samples:
            return {"v": g[idx].mean()}
        best = self._split(B, g, idx)
        if best is None:
            return {"v": g[idx].mean()}
        _, j, k = best
        left, right = idx[B[idx, j] <= k], idx[B[idx, j] > k]
        if len(left) == 0 or len(right) == 0:
            return {"v": g[idx].mean()}
        return {"j": j, "k": k, "l": self._grow(B, g, left, depth - 1),
                "r": self._grow(B, g, right, depth - 1)}

    def _apply(self, B, node, out, idx):
        if "v" in node:
            out[idx] = node["v"]
            return
        m = B[idx, node["j"]] <= node["k"]
        self._apply(B, node["l"], out, idx[m])
        self._apply(B, node["r"], out, idx[~m])

    def fit(self, X, y):
        B = self._bin(X, fit=True)
        self.base_ = y.mean()
        pred = np.full(len(y), self.base_)
        self.trees_, idx = [], np.arange(len(y))
        for _ in range(self.n_estimators):
            tree = self._grow(B, y - pred, idx, self.max_depth)
            step = np.zeros(len(y))
            self._apply(B, tree, step, idx)
            pred += self.lr * step
            self.trees_.append(tree)
        return self

    def predict(self, X):
        B = self._bin(X)
        out = np.full(len(B), self.base_)
        idx = np.arange(len(B))
        for tree in self.trees_:
            step = np.zeros(len(B))
            self._apply(B, tree, step, idx)
            out += self.lr * step
        return out
