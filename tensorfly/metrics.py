"""tensorfly.metrics — like tf.keras.metrics."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _np(x):
    return x._data if isinstance(x, Tensor) else np.asanyarray(x)


class Metric:
    def __init__(self, name="metric"):
        self.name = name

    def update_state(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def reset_state(self):
        pass


class Mean(Metric):
    def __init__(self, name="mean"):
        super().__init__(name)
        self.total = 0.0
        self.count = 0

    def update_state(self, y_true, y_pred=None):
        v = _np(y_true if y_pred is None else y_pred)
        # if loss Tensor scalar
        self.total += float(np.asanyarray(v).mean())
        self.count += 1

    def result(self):
        return self.total / max(1, self.count)

    def reset_state(self):
        self.total = 0.0
        self.count = 0


class Accuracy(Metric):
    def __init__(self, name="accuracy"):
        super().__init__(name)
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):
        yt, yp = _np(y_true), _np(y_pred)
        if yp.ndim > 1 and yp.shape[-1] > 1:
            yp = yp.argmax(axis=-1)
        elif yp.ndim > 1 and yp.shape[-1] == 1:
            yp = (yp.reshape(-1) >= 0.5).astype(int)
        elif np.issubdtype(np.asanyarray(yp).dtype, np.floating):
            # 1-D probabilities (sigmoid) vs integer labels -> threshold
            _yt = np.asanyarray(yt).ravel()
            if np.issubdtype(np.asanyarray(_yt).dtype, np.integer) or set(np.unique(_yt).tolist()) <= {0, 1, 0.0, 1.0}:
                yp = (np.asanyarray(yp).ravel() >= 0.5).astype(int)
        if yt.ndim > 1 and yt.shape[-1] > 1:
            yt = yt.argmax(axis=-1)
        yt = np.asanyarray(yt).ravel()
        yp = np.broadcast_to(np.asanyarray(yp).ravel(), yt.shape)
        self.correct += int((yt == yp).sum())
        self.total += yt.size

    def result(self):
        return self.correct / max(1, self.total)

    def reset_state(self):
        self.correct = 0
        self.total = 0


SparseCategoricalAccuracy = Accuracy
CategoricalAccuracy = Accuracy
BinaryAccuracy = Accuracy


class MeanSquaredError(Metric):
    def __init__(self, name="mse"):
        super().__init__(name)
        self.total, self.count = 0.0, 0

    def update_state(self, y_true, y_pred):
        a, b = _np(y_true).astype(float), _np(y_pred).astype(float)
        self.total += float(((a - b) ** 2).mean())
        self.count += 1

    def result(self):
        return self.total / max(1, self.count)

    def reset_state(self):
        self.total, self.count = 0.0, 0


class MeanAbsoluteError(Metric):
    def __init__(self, name="mae"):
        super().__init__(name)
        self.total, self.count = 0.0, 0

    def update_state(self, y_true, y_pred):
        a, b = _np(y_true).astype(float), _np(y_pred).astype(float)
        self.total += float(np.abs(a - b).mean())
        self.count += 1

    def result(self):
        return self.total / max(1, self.count)

    def reset_state(self):
        self.total, self.count = 0.0, 0


class Precision(Metric):
    def __init__(self, name="precision", threshold=0.5):
        super().__init__(name)
        self.threshold = threshold
        self.tp, self.fp = 0, 0

    def update_state(self, y_true, y_pred):
        yt = _np(y_true).ravel()
        yp = _np(y_pred)
        if yp.ndim > 1 and yp.shape[-1] > 1:
            yp = yp.argmax(-1)
        else:
            yp = (yp.ravel() >= self.threshold).astype(int)
        if yt.ndim > 1 and yt.shape[-1] > 1:
            yt = yt.argmax(-1)
        yt = yt.ravel().astype(int)
        yp = np.broadcast_to(yp.ravel(), yt.shape)
        self.tp += int(((yp == 1) & (yt == 1)).sum())
        self.fp += int(((yp == 1) & (yt != 1)).sum())

    def result(self):
        return self.tp / max(1, self.tp + self.fp)

    def reset_state(self):
        self.tp, self.fp = 0, 0


class Recall(Metric):
    def __init__(self, name="recall", threshold=0.5):
        super().__init__(name)
        self.threshold = threshold
        self.tp, self.fn = 0, 0

    def update_state(self, y_true, y_pred):
        yt = _np(y_true).ravel()
        yp = _np(y_pred)
        if yp.ndim > 1 and yp.shape[-1] > 1:
            yp = yp.argmax(-1)
        else:
            yp = (yp.ravel() >= self.threshold).astype(int)
        if yt.ndim > 1 and yt.shape[-1] > 1:
            yt = yt.argmax(-1)
        yt = yt.ravel().astype(int)
        yp = np.broadcast_to(yp.ravel(), yt.shape)
        self.tp += int(((yp == 1) & (yt == 1)).sum())
        self.fn += int(((yp != 1) & (yt == 1)).sum())

    def result(self):
        return self.tp / max(1, self.tp + self.fn)

    def reset_state(self):
        self.tp, self.fn = 0, 0


class F1Score(Metric):
    def __init__(self, name="f1", threshold=0.5):
        super().__init__(name)
        self._p, self._r = Precision(threshold=threshold), Recall(threshold=threshold)

    def update_state(self, y_true, y_pred):
        self._p.update_state(y_true, y_pred)
        self._r.update_state(y_true, y_pred)

    def result(self):
        p, r = self._p.result(), self._r.result()
        return 2 * p * r / max(1e-12, p + r)

    def reset_state(self):
        self._p.reset_state()
        self._r.reset_state()


class AUC(Metric):
    """Binary ROC-AUC via the Mann-Whitney rank statistic (handles ties)."""

    def __init__(self, name="auc"):
        super().__init__(name)
        self._y, self._p = [], []

    def update_state(self, y_true, y_pred):
        yt = _np(y_true).ravel()
        yp = _np(y_pred)
        if yp.ndim > 1:
            yp = yp[:, 1] if yp.shape[1] == 2 else yp.ravel()
        self._y.append(np.asanyarray(yt).ravel())
        self._p.append(np.asanyarray(yp, dtype=float).ravel())

    def result(self):
        y = np.concatenate(self._y) if self._y else np.array([])
        p = np.concatenate(self._p) if self._p else np.array([])
        if not len(y):
            return 0.0
        y = np.asanyarray(y).ravel()
        p = np.asanyarray(p, dtype=float).ravel()
        P, N = int((y == 1).sum()), int((y != 1).sum())
        if not P or not N:
            return 0.0
        # Mann-Whitney rank AUC (correct with ties)
        order = np.argsort(p, kind="stable")
        ranks = np.empty(len(p))
        # average ranks for tied groups
        i = 0
        sorted_p = p[order]
        while i < len(p):
            j = i
            while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-based ranks
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        rank_sum_pos = float(ranks[y == 1].sum())
        return float((rank_sum_pos - P * (P + 1) / 2) / (P * N))

    def reset_state(self):
        self._y, self._p = [], []


def _resolve_metric(m):
    if isinstance(m, Metric):
        return m
    table = {"accuracy": Accuracy, "acc": Accuracy,
             "mse": MeanSquaredError, "mae": MeanAbsoluteError,
             "precision": Precision, "recall": Recall, "f1": F1Score, "auc": AUC,
             "mean": Mean}
    if isinstance(m, str) and m.lower() in table:
        return table[m.lower()]()
    raise ValueError(f"unknown metric '{m}'")
