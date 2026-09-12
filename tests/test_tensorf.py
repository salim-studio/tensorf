import numpy as np
import tensorf as tf


def test_constant_and_ops():
    a = tf.constant([1., 2., 3.])
    assert a.shape == (3,)
    assert str(a.dtype) == "float32"
    assert np.allclose((a * 2).numpy(), [2, 4, 6])
    assert np.allclose(tf.matmul(tf.ones((2, 3)), tf.ones((3, 4))).numpy(), np.full((2, 4), 3))


def test_gradient_tape():
    with tf.GradientTape() as tape:
        x = tf.Variable([2.0, 3.0])
        y = tf.reduce_sum(x * x)
    g = tape.gradient(y, x)
    assert np.allclose(g.numpy(), [4, 6])


def test_broadcast_grad():
    with tf.GradientTape() as tape:
        W = tf.Variable(np.ones((3, 2), np.float32))
        b = tf.Variable(np.zeros(2, np.float32))
        x = tf.constant(np.ones((5, 3), np.float32))
        y = tf.reduce_sum(tf.matmul(x, W) + b)
    gw, gb = tape.gradient(y, [W, b])
    assert gw.numpy().shape == (3, 2)
    assert gb.numpy().shape == (2,)
    assert np.allclose(gb.numpy(), [5, 5])


def test_softmax_xent():
    with tf.GradientTape() as t:
        W = tf.Variable(np.random.randn(4, 3).astype(np.float32))
        x = tf.constant(np.random.randn(6, 4).astype(np.float32))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy(tf.matmul(x, W), [0, 1, 2, 0, 1, 2]))
    g = t.gradient(loss, W)
    assert g.numpy().shape == (4, 3)
    assert np.isfinite(g.numpy()).all()


def test_conv2d():
    x = tf.constant(np.random.randn(2, 8, 8, 3).astype(np.float32))
    f = tf.Variable(np.random.randn(3, 3, 3, 4).astype(np.float32))
    y = tf.conv2d(x, f, padding="SAME")
    assert y.shape == (2, 8, 8, 4)


def test_sequential_fit_reduces_loss():
    np.random.seed(0)
    X = np.random.randn(200, 4).astype(np.float32)
    Y = (X.sum(1, keepdims=True) > 0).astype(np.float32)
    m = tf.keras.Sequential([tf.keras.layers.Dense(16, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    h = m.fit(X, Y, epochs=3, batch_size=32, verbose=0)
    assert h["loss"][-1] < h["loss"][0]


def test_dataset():
    X = np.arange(20).reshape(10, 2)
    ds = tf.data.Dataset.from_tensor_slices(X).batch(4)
    batches = list(ds)
    assert len(batches) == 3
    assert batches[0].shape == (4, 2)


def test_function_and_metrics():
    @tf.function
    def f(a, b):
        return tf.matmul(a, b)

    A = tf.constant(np.eye(3, dtype=np.float32))
    assert np.allclose(f(A, A).numpy(), np.eye(3))
    m = tf.metrics.Accuracy()
    m.update_state([0, 1, 1], [0, 0, 1])
    assert abs(m.result() - 2 / 3) < 1e-6
