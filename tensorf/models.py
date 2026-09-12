"""tensorf.models — Sequential + Model (like tf.keras)."""
from __future__ import annotations

import pickle

import numpy as np

from .tensor import Tensor, Variable, convert_to_tensor
from .layers import Layer
from .autodiff import GradientTape


def _np(x):
    return x._data if isinstance(x, Tensor) else np.asanyarray(x)


class Sequential:
    def __init__(self, layers=None, name="sequential"):
        self.layers: list[Layer] = list(layers) if layers else []
        self.name = name
        self.optimizer = None
        self.loss_fn = None
        self.metrics_list = []
        self._training = True

    def add(self, layer):
        self.layers.append(layer)
        return self

    def __call__(self, x, training=None):
        t = x if isinstance(x, Tensor) else convert_to_tensor(x, dtype=np.float32)
        for l in self.layers:
            l._training = self._training if training is None else training
            t = l(t, training=l._training)
        return t

    @property
    def trainable_variables(self):
        vs = []
        for l in self.layers:
            vs.extend(l.trainable_variables)
        return vs

    @property
    def weights(self):
        return self.trainable_variables

    def get_weights(self):
        return [v.numpy().copy() for v in self.trainable_variables]

    def set_weights(self, weights):
        for v, w in zip(self.trainable_variables, weights):
            v.assign(w)

    def summary(self, print_fn=print):
        lines = [f"Model: {self.name}", "-" * 60]
        total = 0
        for l in self.layers:
            n = sum(int(np.prod(v.shape)) for v in l.trainable_variables)
            total += n
            lines.append(f" {l.name:25s} {l.__class__.__name__:20s} params: {n}")
        lines.append(f"Total params: {total}")
        print_fn("\n".join(lines))

    def compile(self, optimizer="sgd", loss="mse", metrics=None):
        from . import optimizers as _opt, losses as _loss
        if isinstance(optimizer, str):
            optimizer = {"sgd": _opt.SGD, "adam": _opt.Adam, "adamw": _opt.AdamW,
                         "rmsprop": _opt.RMSprop, "adagrad": _opt.Adagrad,
                         "adadelta": _opt.Adadelta, "lion": _opt.Lion}.get(optimizer, _opt.SGD)()
        self.optimizer = optimizer
        if isinstance(loss, str):
            self.loss_fn = {"mse": _loss.mean_squared_error, "mae": _loss.mean_absolute_error,
                            "binary_crossentropy": _loss.binary_crossentropy,
                            "categorical_crossentropy": _loss.categorical_crossentropy,
                            "sparse_categorical_crossentropy": _loss.sparse_categorical_crossentropy,
                            "huber": _loss.huber, "hinge": _loss.hinge,
                            }.get(loss, _loss.mean_squared_error)
            self._loss_name = loss
        else:
            self.loss_fn = loss
            self._loss_name = getattr(loss, "__name__", "loss")
        from .metrics import _resolve_metric
        self.metrics_list = [(_resolve_metric(m) if isinstance(m, str) else m)
                             for m in (metrics or [])]
        self._metric_names = [m.name for m in self.metrics_list]

    def _loss_scalar(self, y_true, y_pred) -> Tensor:
        return self.loss_fn(y_true, y_pred)

    def fit(self, x, y=None, epochs=1, batch_size=32, verbose=1, shuffle=True,
            validation_data=None, validation_split=0.0, callbacks=None, class_weight=None):
        from .data import Dataset
        from .metrics import Metric
        X = _np(x)
        Y = _np(y) if y is not None else None
        # validation_split
        if validation_split and validation_data is None:
            n = len(X)
            nv = int(n * validation_split)
            idx = np.arange(n)
            np.random.shuffle(idx)
            vi, ti = idx[:nv], idx[nv:]
            validation_data = (X[vi], Y[vi] if Y is not None else None)
            X, Y = X[ti], (Y[ti] if Y is not None else None)
        n = len(X)
        cbs = list(callbacks or [])
        from .callbacks import History
        hist_cb = History()
        cbs = [hist_cb] + cbs
        hist = {"loss": []}
        for m in self.metrics_list:
            hist[m.name] = []
            hist[f"val_{m.name}"] = []
        for cb in cbs:
            cb.on_train_begin()
        for ep in range(epochs):
            for cb in cbs:
                cb.on_epoch_begin(ep)
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            total, nb = 0.0, 0
            for m in self.metrics_list:
                m.reset_state()
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                xb = convert_to_tensor(X[bi].astype(np.float32))
                yb = convert_to_tensor(Y[bi].astype(np.float32)) if Y is not None else None
                with GradientTape() as tape:
                    pred = self(xb, training=True)
                    loss = self._loss_scalar(yb, pred)
                grads = tape.gradient(loss, self.trainable_variables)
                # NOTE: class_weight is accepted for API compatibility (reserved).
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                total += float(loss._data)
                nb += 1
                if Y is not None:
                    for m in self.metrics_list:
                        try:
                            m.update_state(Y[bi], pred._data)
                        except Exception:
                            pass
                for cb in cbs:
                    cb.on_batch_end(nb, {"loss": float(loss._data)})
            avg = total / max(1, nb)
            hist["loss"].append(avg)
            logs = {"loss": avg, "_model": self}
            for m in self.metrics_list:
                try:
                    logs[m.name] = hist[m.name].append(float(m.result())) or float(m.result())
                except Exception:
                    pass
            if validation_data is not None:
                vx, vy = validation_data
                vp = self(convert_to_tensor(_np(vx).astype(np.float32)), training=False)
                vl = self._loss_scalar(convert_to_tensor(_np(vy).astype(np.float32)), vp) if vy is not None else None
                if vl is not None:
                    hist.setdefault("val_loss", []).append(float(vl._data))
                    logs["val_loss"] = float(vl._data)
                for m in self.metrics_list:
                    try:
                        m.reset_state()
                        m.update_state(_np(vy), vp._data)
                        v = float(m.result())
                        hist[f"val_{m.name}"].append(v)
                        logs[f"val_{m.name}"] = v
                    except Exception:
                        pass
                if verbose:
                    msg = f"epoch {ep+1}/{epochs} - loss: {avg:.4f} - val_loss: {logs.get('val_loss', float('nan')):.4f}"
                    for m in self.metrics_list:
                        msg += f" - {m.name}: {logs.get(m.name, float('nan')):.4f} - val_{m.name}: {logs.get('val_'+m.name, float('nan')):.4f}"
                    print(msg)
            elif verbose:
                msg = f"epoch {ep+1}/{epochs} - loss: {avg:.4f}"
                for m in self.metrics_list:
                    msg += f" - {m.name}: {logs.get(m.name, float('nan')):.4f}"
                print(msg)
            stop = False
            for cb in cbs:
                cb.on_epoch_end(ep, logs)
                if getattr(cb, "stop_training", False):
                    stop = True
            if stop:
                if verbose:
                    print(f"Early stopping at epoch {ep+1}")
                break
        for cb in cbs:
            cb.on_train_end(hist)
        # merge the history callback
        for k, v in hist_cb.history.items():
            if k.startswith("_"):
                continue
            if k not in hist:
                hist[k] = v
        self.history = hist
        return hist

    def evaluate(self, x, y, batch_size=256):
        X, Y = _np(x).astype(np.float32), _np(y)
        pred = self(convert_to_tensor(X), training=False)
        loss = self._loss_scalar(convert_to_tensor(Y.astype(np.float32)), pred)
        out = {"loss": float(loss._data)}
        for m in self.metrics_list:
            try:
                m.reset_state()
                m.update_state(Y, pred._data)
                out[m.name] = float(m.result())
            except Exception:
                pass
        return out if len(out) > 1 else out["loss"]

    def predict(self, x, batch_size=256):
        X = _np(x).astype(np.float32)
        outs = []
        for s in range(0, len(X), batch_size):
            p = self(convert_to_tensor(X[s:s + batch_size]), training=False)
            outs.append(p._data)
        return np.concatenate(outs)

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_weights(), f)

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_weights(pickle.load(f))

    def save(self, path):
        """Save the full model (config + weights)."""
        cfg = {"class": self.__class__.__name__, "name": self.name,
               "layers": [{"class": l.__class__.__name__, "name": l.name,
                           "params": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                      for k, v in vars(l).items()
                                      if k.startswith("_") is False and isinstance(v, (int, float, str, bool, list, tuple))}}
                          for l in getattr(self, "layers", [])],
               "loss": getattr(self, "_loss_name", "mse"),
               "weights": [w.tolist() for w in self.get_weights()]}
        with open(path, "wb") as f:
            pickle.dump(cfg, f)
        return path


class Model(Sequential):
    """Subclassing API: class MyModel(tensorf.Model): def call(...)"""

    def call(self, x, training=True):
        raise NotImplementedError

    def __call__(self, x, training=None):
        t = x if isinstance(x, Tensor) else convert_to_tensor(np.asanyarray(x), dtype=np.float32)
        # build lazily not needed for subclass
        return self.call(t, training=self._training if training is None else training)
