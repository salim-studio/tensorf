"""tensorfly quickstart: autograd + a tiny neural net (runs with numpy only)."""
import numpy as np

import tensorfly as tf

print("tensorfly", tf.__version__, tf.info())

# 1. Autograd, TF-style
x = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x * x)
print("gradient:", tape.gradient(y, x).numpy())  # [4. 6.]

# 2. Binary classification with Keras-style API
rng = np.random.default_rng(0)
X = rng.standard_normal((400, 4)).astype(np.float32)
Y = (X.sum(axis=1, keepdims=True) > 0).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=5, batch_size=32, validation_split=0.2)
model.summary()
print("eval:", model.evaluate(X, Y))
