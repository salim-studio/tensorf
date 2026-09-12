"""tensorf speed comparison: fusion + parallelism (fair in-library comparisons)."""
import time
import numpy as np
import tensorf as tf


def bench(name, fn, n=5):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{name:42s} {dt:8.2f} ms")
    return dt


if __name__ == "__main__":
    # 1) fused dense vs separate ops (same library — shows temporary savings)
    X = tf.constant(np.random.randn(4096, 512).astype(np.float32))
    W = tf.constant(np.random.randn(512, 256).astype(np.float32))
    B = tf.constant(np.zeros(256, np.float32))
    t_fused = bench("dense fused (1 pass)", lambda: tf.nn.dense_fused(X, W, B, "relu").numpy())
    t_sep = bench("dense separate (matmul+bias+relu)", lambda: tf.relu(tf.matmul(X, W) + B).numpy())
    print(f"  fusion speedup: {t_sep / max(t_fused, 1e-9):.2f}x")

    # 2) parallelism on/off for a large element-wise op (positive values for sqrt)
    big = tf.constant((np.random.randn(4_000_000).astype(np.float32) ** 2 + 0.5))
    tf.set_workers(4)
    t_par = bench("element-wise sqrt (4 workers)", lambda: tf.sqrt(big).numpy())
    tf.set_workers(1)
    t_1 = bench("element-wise sqrt (1 worker)", lambda: tf.sqrt(big).numpy())
    tf.set_workers(4)
    print(f"  parallel speedup: {t_1 / max(t_par, 1e-9):.2f}x")

    # 3) full training step (fused Adam + fused dense)
    np.random.seed(0)
    Xd = np.random.randn(512, 32).astype(np.float32)
    Yd = (Xd.sum(1, keepdims=True) > 0).astype(np.float32)
    m = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy")

    def step():
        xb = tf.constant(Xd)
        yb = tf.constant(Yd)
        with tf.GradientTape() as tape:
            loss = m.loss_fn(yb, m(xb, training=True))
        m.optimizer.apply_gradients(zip(tape.gradient(loss, m.trainable_variables),
                                        m.trainable_variables))
        return float(loss.numpy() if hasattr(loss, "numpy") else loss)

    t_step = bench("full train step (512 samples)", step, n=3)
    print(f"  throughput: {512 / (t_step / 1000):.0f} samples/s")
