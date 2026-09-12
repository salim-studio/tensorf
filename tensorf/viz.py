"""tensorf.viz — quick plots (optional matplotlib)."""
from __future__ import annotations


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("tf.viz needs matplotlib: pip install 'tensorf[viz]'") from e
    return plt


def plot_history(history, save_path=None, show=False):
    plt = _plt()
    h = history.history if hasattr(history, "history") else history
    for k, v in h.items():
        if not k.startswith("val_"):
            plt.plot(v, label=k)
            if f"val_{k}" in h:
                plt.plot(h[f"val_{k}"], "--", label=f"val_{k}")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close() if save_path else None
    return plt.gcf()


def plot_confusion_matrix(cm, labels=None, save_path=None, show=False):
    import numpy as np
    plt = _plt()
    cm = np.asanyarray(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cm.shape[1]), labels or range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]), labels or range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def scatter(X, y=None, save_path=None, show=False):
    import numpy as np
    plt = _plt()
    A = X._data if hasattr(X, "_data") else np.asanyarray(X)
    if A.shape[1] >= 2:
        if y is None:
            plt.scatter(A[:, 0], A[:, 1], s=8)
        else:
            lab = np.asanyarray(y._data if hasattr(y, "_data") else y).ravel()
            for c in np.unique(lab):
                m = lab == c
                plt.scatter(A[m, 0], A[m, 1], s=8, label=str(c))
            plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gcf()


__all__ = ["plot_history", "plot_confusion_matrix", "scatter"]
