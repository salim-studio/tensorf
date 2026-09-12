"""tensorfly.callbacks — like tf.keras.callbacks."""
from __future__ import annotations

import csv
import os

import numpy as np


class Callback:
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass


class History(Callback):
    def __init__(self):
        self.history: dict[str, list] = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            if k.startswith("_"):
                continue
            try:
                self.history.setdefault(k, []).append(float(v))
            except Exception:
                pass


class LambdaCallback(Callback):
    def __init__(self, on_epoch_end=None):
        self._fn = on_epoch_end
    def on_epoch_end(self, epoch, logs=None):
        if self._fn:
            self._fn(epoch, logs)


class EarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.restore = restore_best_weights
        self.best, self.wait, self.best_w = np.inf, 0, None
        self.stopped_epoch = 0
        self.stop_training = False
    def on_train_begin(self, logs=None):
        self.best, self.wait, self.stop_training = np.inf, 0, False
    def on_epoch_end(self, epoch, logs=None):
        cur = (logs or {}).get(self.monitor)
        if cur is None:
            return
        if cur < self.best - self.min_delta:
            self.best, self.wait = cur, 0
            m = (logs or {}).get("_model")
            if m is not None and self.restore:
                self.best_w = m.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                m = (logs or {}).get("_model")
                if m is not None and self.restore and self.best_w is not None:
                    m.set_weights(self.best_w)


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor="val_loss", save_best_only=True, save_weights_only=True):
        self.filepath, self.monitor = filepath, monitor
        self.save_best_only, self.weights_only = save_best_only, save_weights_only
        self.best = np.inf
    def on_epoch_end(self, epoch, logs=None):
        m = (logs or {}).get("_model")
        if m is None:
            return
        cur = (logs or {}).get(self.monitor, np.inf)
        if not self.save_best_only or cur < self.best:
            self.best = cur
            if self.weights_only:
                m.save_weights(self.filepath)
            else:
                m.save(self.filepath)


class ReduceLROnPlateau(Callback):
    def __init__(self, monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1):
        self.monitor, self.factor, self.patience, self.min_lr = monitor, factor, patience, min_lr
        self.verbose = verbose
        self.best, self.wait = np.inf, 0
    def on_epoch_end(self, epoch, logs=None):
        m = (logs or {}).get("_model")
        cur = (logs or {}).get(self.monitor)
        if cur is None or m is None or m.optimizer is None:
            return
        if cur < self.best:
            self.best, self.wait = cur, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old = float(m.optimizer.learning_rate if not callable(m.optimizer.learning_rate) else m.optimizer._lr())
                new = max(self.min_lr, old * self.factor)
                m.optimizer.learning_rate = new
                if self.verbose:
                    print(f"ReduceLROnPlateau: lr {old:.6f} -> {new:.6f}")
                self.wait = 0


class CSVLogger(Callback):
    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self._f = None
        self._w = None
    def on_train_begin(self, logs=None):
        self._f = open(self.filename, "a" if self.append else "w", newline="")
        self._w = None
    def on_epoch_end(self, epoch, logs=None):
        logs = {k: v for k, v in (logs or {}).items() if not k.startswith("_")}
        if self._w is None:
            self._w = csv.DictWriter(self._f, fieldnames=["epoch"] + list(logs.keys()))
            self._w.writeheader()
        self._w.writerow({"epoch": epoch, **{k: float(v) for k, v in logs.items()}})
        self._f.flush()
    def on_train_end(self, logs=None):
        if self._f:
            self._f.close()


__all__ = ["Callback", "History", "LambdaCallback", "EarlyStopping",
           "ModelCheckpoint", "ReduceLROnPlateau", "CSVLogger"]
