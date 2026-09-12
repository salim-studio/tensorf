"""tensorfly.ml — classical machine learning (sklearn-like, numpy only).

    import tensorfly as tf
    clf = tf.ml.LogisticRegression().fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    km = tf.ml.KMeans(k=3).fit(X)
    X2 = tf.ml.PCA(n_components=2).fit_transform(X)

Models: LinearRegression, Ridge, LogisticRegression, KNNClassifier,
KNNRegressor, GaussianNB, KMeans, PCA, DecisionTreeClassifier (simple CART).
"""
from __future__ import annotations

import numpy as np


def _X(x):
    return x._data.astype(np.float64) if hasattr(x, "_data") else np.asanyarray(x, dtype=np.float64)


def _y(x):
    return np.asanyarray(x._data if hasattr(x, "_data") else x)


def accuracy_score(y_true, y_pred):
    a, b = _y(y_true).ravel(), _y(y_pred).ravel()
    return float((a == b).mean())


def mean_squared_error(y_true, y_pred):
    a, b = _X(y_true).ravel(), _X(y_pred).ravel()
    return float(((a - b) ** 2).mean())


def mean_absolute_error(y_true, y_pred):
    a, b = _X(y_true).ravel(), _X(y_pred).ravel()
    return float(np.abs(a - b).mean())


def r2_score(y_true, y_pred):
    a, b = _X(y_true).ravel(), _X(y_pred).ravel()
    ss = ((a - b) ** 2).sum()
    tot = ((a - a.mean()) ** 2).sum() + 1e-12
    return float(1 - ss / tot)


def confusion_matrix(y_true, y_pred):
    a, b = _y(y_true).ravel(), _y(y_pred).ravel()
    labels = np.unique(np.concatenate([a, b]))
    m = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), np.int64)
    for t, p in zip(a, b):
        cm[m[t], m[p]] += 1
    return cm, labels


def classification_report(y_true, y_pred) -> dict:
    cm, labels = confusion_matrix(y_true, y_pred)
    rep = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        prec = tp / max(1, cm[:, i].sum())
        rec = tp / max(1, cm[i, :].sum())
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        rep[int(l) if isinstance(l, (int, np.integer)) else l] = {
            "precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    rep["accuracy"] = accuracy_score(y_true, y_pred)
    return rep


class LinearRegression:
    """Closed-form solution (lstsq) — fast and exact."""
    def fit(self, X, y):
        A = np.concatenate([_X(X), np.ones((len(_X(X)), 1))], axis=1)
        b = _y(y).astype(np.float64).ravel()
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.coef_, self.intercept_ = w[:-1].astype(np.float32), float(w[-1])
        return self
    def predict(self, X):
        return (_X(X) @ self.coef_.astype(np.float64) + self.intercept_).astype(np.float32)
    def score(self, X, y):
        return r2_score(y, self.predict(X))


class Ridge(LinearRegression):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        A = _X(X)
        n, d = A.shape
        b = _y(y).astype(np.float64).ravel()
        M = A.T @ A + self.alpha * np.eye(d)
        v = A.T @ b
        self.coef_ = np.linalg.solve(M, v).astype(np.float32)
        self.intercept_ = float(b.mean() - A.mean(0) @ self.coef_.astype(np.float64))
        # refit intercept properly: center
        mu_x, mu_y = A.mean(0), b.mean()
        self.coef_ = np.linalg.solve(M, A.T @ (b - mu_y) + self.alpha * 0).astype(np.float32)
        # simpler: solve augmented with unpenalized intercept via centering
        Ac = A - mu_x
        bc = b - mu_y
        self.coef_ = np.linalg.solve(Ac.T @ Ac + self.alpha * np.eye(d), Ac.T @ bc).astype(np.float32)
        self.intercept_ = float(mu_y - mu_x @ self.coef_.astype(np.float64))
        return self


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500, l2=1e-4, batch_size=None, seed=0, verbose=False):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.batch_size, self.seed, self.verbose = batch_size, seed, verbose
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        A = _X(X)
        b = _y(y).astype(np.float64).ravel()
        classes = np.unique(b)
        self.classes_ = classes
        n, d = A.shape
        if len(classes) == 2:
            t = (b == classes[1]).astype(np.float64)
            w = np.zeros(d)
            bo = 0.0
            bs = self.batch_size or n
            for ep in range(self.epochs):
                idx = rng.permutation(n)
                for s in range(0, n, bs):
                    bi = idx[s:s + bs]
                    p = 1 / (1 + np.exp(-(A[bi] @ w + bo)))
                    err = p - t[bi]
                    w -= self.lr * (A[bi].T @ err / len(bi) + self.l2 * w)
                    bo -= self.lr * err.mean()
                if self.verbose and ep % 100 == 0:
                    print(f"ep {ep} loss={float(-(t*np.log(p+1e-9)+(1-t)*np.log(1-p+1e-9)).mean()):.4f}")
            self.coef_, self.intercept_ = w.astype(np.float32), float(bo)
        else:
            # softmax multinomial (batch GD)
            K = len(classes)
            W = np.zeros((d, K))
            bb = np.zeros(K)
            Y = np.zeros((n, K))
            for k, c in enumerate(classes):
                Y[:, k] = (b == c)
            bs = self.batch_size or n
            for _ in range(self.epochs):
                idx = rng.permutation(n)
                for s in range(0, n, bs):
                    bi = idx[s:s + bs]
                    Z = A[bi] @ W + bb
                    Z -= Z.max(1, keepdims=True)
                    P = np.exp(Z)
                    P /= P.sum(1, keepdims=True)
                    E = (P - Y[bi]) / len(bi)
                    W -= self.lr * (A[bi].T @ E + self.l2 * W)
                    bb -= self.lr * E.sum(0)
            self.coef_, self.intercept_ = W.astype(np.float32), bb.astype(np.float32)
        return self
    def predict_proba(self, X):
        A = _X(X)
        if len(self.classes_) == 2:
            p1 = 1 / (1 + np.exp(-(A @ self.coef_ + self.intercept_)))
            return np.stack([1 - p1, p1], 1).astype(np.float32)
        Z = A @ self.coef_ + self.intercept_
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        return (P / P.sum(1, keepdims=True)).astype(np.float32)
    def predict(self, X):
        P = self.predict_proba(X)
        return np.asanyarray(self.classes_)[P.argmax(1)]
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_, self.y_ = _X(X), _y(y).ravel()
        return self
    def predict(self, X):
        A = _X(X)
        d2 = ((A[:, None, :] - self.X_[None]) ** 2).sum(-1)
        idx = np.argpartition(d2, min(self.k, len(self.X_) - 1), axis=1)[:, :self.k]
        out = []
        for row in idx:
            vals, c = np.unique(self.y_[row], return_counts=True)
            out.append(vals[c.argmax()])
        return np.array(out)


class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_, self.y_ = _X(X), _y(y).astype(np.float64).ravel()
        return self
    def predict(self, X):
        A = _X(X)
        d2 = ((A[:, None, :] - self.X_[None]) ** 2).sum(-1)
        idx = np.argpartition(d2, min(self.k, len(self.X_) - 1), axis=1)[:, :self.k]
        return np.array([self.y_[row].mean() for row in idx], np.float32)


class GaussianNB:
    def fit(self, X, y):
        A, b = _X(X), _y(y).ravel()
        self.classes_ = np.unique(b)
        self.theta_, self.var_, self.prior_ = [], [], []
        for c in self.classes_:
            Ac = A[b == c]
            self.theta_.append(Ac.mean(0))
            self.var_.append(Ac.var(0) + 1e-9)
            self.prior_.append(len(Ac) / len(A))
        self.theta_, self.var_ = np.stack(self.theta_), np.stack(self.var_)
        self.prior_ = np.array(self.prior_)
        return self
    def predict(self, X):
        A = _X(X)
        ll = -0.5 * (((A[:, None, :] - self.theta_[None]) ** 2 / self.var_[None]).sum(-1)
                     + np.log(2 * np.pi * self.var_).sum(-1)[None, :]) + np.log(self.prior_)[None, :]
        return self.classes_[ll.argmax(1)]


class KMeans:
    def __init__(self, k=3, max_iter=100, seed=0, n_init=3):
        self.k, self.max_iter, self.seed, self.n_init = k, max_iter, seed, n_init
    def fit(self, X):
        A = _X(X).astype(np.float64)
        best = None
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_init):
            C = A[rng.choice(len(A), self.k, replace=False)]
            for _ in range(self.max_iter):
                d2 = ((A[:, None, :] - C[None]) ** 2).sum(-1)
                lab = d2.argmin(1)
                Cn = np.stack([A[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(self.k)])
                if np.allclose(Cn, C):
                    C = Cn
                    break
                C = Cn
            inertia = float(((A - C[lab]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, C, lab)
        self.inertia_, self.cluster_centers_, self.labels_ = best[0], best[1].astype(np.float32), best[2]
        return self
    def predict(self, X):
        A = _X(X)
        return ((A[:, None, :] - self.cluster_centers_[None].astype(np.float64)) ** 2).sum(-1).argmin(1)


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X):
        A = _X(X)
        self.mean_ = A.mean(0)
        _, S, Vt = np.linalg.svd(A - self.mean_, full_matrices=False)
        self.components_ = Vt[:self.n_components].astype(np.float32)
        var = (S ** 2) / max(1, len(A) - 1)
        self.explained_variance_ratio_ = (var / var.sum())[:self.n_components]
        return self
    def transform(self, X):
        return (( _X(X) - self.mean_) @ self.components_.T).astype(np.float32)
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, Z):
        return (np.asanyarray(Z, dtype=np.float64) @ self.components_.astype(np.float64) + self.mean_).astype(np.float32)


class _TreeNode:
    __slots__ = ("feat", "thr", "left", "right", "value")
    def __init__(self, value):
        self.feat = None; self.thr = None; self.left = None; self.right = None; self.value = value


class DecisionTreeClassifier:
    """Simple binary CART (gini) — for teaching and small/medium data."""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
    def fit(self, X, y):
        A, b = _X(X), _y(y).ravel()
        self.classes_ = np.unique(b)
        self.root_ = self._build(A, b, 0)
        return self
    def _gini(self, b):
        if not len(b):
            return 0
        _, c = np.unique(b, return_counts=True)
        p = c / c.sum()
        return 1 - (p ** 2).sum()
    def _build(self, A, b, depth):
        vals, c = np.unique(b, return_counts=True)
        node = _TreeNode(vals[c.argmax()])
        if depth >= self.max_depth or len(b) < self.min_samples_split or len(vals) == 1:
            return node
        best = (1e9, None, None)
        g0 = self._gini(b)
        for f in range(A.shape[1]):
            thrs = np.unique(A[:, f])
            if len(thrs) > 32:
                thrs = np.percentile(thrs, np.linspace(0, 100, 33))
            for t in thrs:
                L, R = b[A[:, f] <= t], b[A[:, f] > t]
                if not len(L) or not len(R):
                    continue
                g = len(L) / len(b) * self._gini(L) + len(R) / len(b) * self._gini(R)
                if g < best[0]:
                    best = (g, f, t)
        if best[1] is None or best[0] >= g0:
            return node
        node.feat, node.thr = best[1], best[2]
        node.left = self._build(A[A[:, node.feat] <= node.thr], b[A[:, node.feat] <= node.thr], depth + 1)
        node.right = self._build(A[A[:, node.feat] > node.thr], b[A[:, node.feat] > node.thr], depth + 1)
        return node
    def predict(self, X):
        A = _X(X)
        out = []
        for row in A:
            n = self.root_
            while n.feat is not None:
                n = n.left if row[n.feat] <= n.thr else n.right
            out.append(n.value)
        return np.array(out)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


__all__ = ["accuracy_score", "mean_squared_error", "mean_absolute_error", "r2_score",
           "confusion_matrix", "classification_report", "LinearRegression", "Ridge",
           "LogisticRegression", "KNNClassifier", "KNNRegressor", "GaussianNB",
           "KMeans", "PCA", "DecisionTreeClassifier"]
