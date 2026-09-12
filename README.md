<p align="center">
  <img src="https://raw.githubusercontent.com/salim-studio/tensorf/main/assets/banner.svg" alt="tensorf banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/salim-studio/tensorf"><img src="https://img.shields.io/badge/repo-salim--studio%2Ftensorf-4F46E5?logo=github" alt="GitHub repo"></a>
  <img src="https://img.shields.io/badge/version-0.2.0-0891B2" alt="version">
  <img src="https://img.shields.io/badge/python-%3E%3D3.9-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/dependencies-numpy_only-22C55E" alt="numpy only">
  <img src="https://img.shields.io/badge/license-MIT-8B5CF6" alt="license">
  <img src="https://img.shields.io/badge/tests-15_passing-16A34A" alt="tests">
</p>

# 🦋 tensorf

**TensorFlow-compatible machine learning, minus the weight.** `tensorf` mirrors the `tensorflow` / `tf.keras` API — same function names, same model code — but runs on a NumPy-only, CPU-tuned engine. No 500&nbsp;MB install, no CUDA required to get started.

And it goes further: tensorf ships a full **data stack** (SQL databases, DataFrames, I/O, stats, preprocessing, classical ML, NLP utilities), so one tiny library takes you from raw CSV to trained model.

```python
import tensorf as tf

x = tf.constant([[1., 2.], [3., 4.]])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x * x)
print(tape.gradient(y, x))  # just like tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64)
```

## Why tensorf?

| Audience | What you get |
|---|---|
| **Developers** | Drop-in TF/Keras-style API, `tf.io` (CSV/JSON/NPY/Parquet), `tf.db` (SQLite/DuckDB), `tf.function`, model save/load |
| **Data analysts** | `tf.DataFrame` (`head/describe/groupby/merge/corr`), `tf.stats`, `tf.viz`, SQL → DataFrame in one call |
| **Data scientists** | `tf.preprocessing` (scalers, encoders, imputation, splits), `tf.ml` (LogReg, KMeans, PCA, trees, KNN, Naive Bayes + reports), `tf.text` (tokenizer, TF-IDF) |
| **ML/DL engineers** | Dense/Conv/RNN/LSTM/GRU/Multi-Head Attention, AdamW/Lion/Adadelta, LR schedules, callbacks, F1/AUC metrics |

## ⚡ Why faster on CPU?

| Technique | Measured effect |
|---|---|
| Fused `dense` (matmul + bias + activation in one buffer, inference) | **1.22×** vs 3 separate ops |
| Parallel element-wise ops (thread pool) for arrays > 262k elements | **1.2–1.7×** vs single thread |
| `float32` + contiguous arrays by default | Less memory, better SIMD |
| Fused in-place Adam step | One memory pass per variable |
| `tf.function` fast path after first trace | No repeated validation overhead |
| `Dataset.prefetch` on a background thread | Loading overlaps training |
| `einsum(optimize=True)`, direct BLAS for large matmuls | Zero overhead over NumPy/BLAS |

> Honest note: for a single tiny op, performance ≈ raw NumPy (~1×) — NumPy itself is optimized C. The wins show up in **fused pipelines + parallelism + zero framework overhead**. Run `python benchmarks/bench.py` to reproduce.

## 📦 Installation

```bash
pip install tensorf            # core: numpy only
pip install "tensorf[io]"      # + pandas, pyarrow (Parquet)
pip install "tensorf[db]"      # + duckdb, sqlalchemy
pip install "tensorf[viz]"     # + matplotlib
pip install "tensorf[all]"     # everything
```

Requires Python ≥ 3.9.

## 🚀 Quickstart

**Deep learning (Keras-style):**
```python
import tensorf as tf

m = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
m.compile(optimizer="adamw", loss="binary_crossentropy", metrics=["accuracy", "f1"])
m.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.15,
      callbacks=[tf.callbacks.EarlyStopping(patience=3)])
print(m.evaluate(X_test, y_test))
```

**Databases → training in 3 lines:**
```python
db = tf.db.Database("app.db")
t = db.read_sql("SELECT age, income FROM users WHERE age > 18")  # -> Tensor
df = db.read_dataframe("SELECT * FROM users")                    # -> DataFrame
```

**Analysis + classical ML:**
```python
df = tf.DataFrame({"age": [20, 30, 25], "city": ["oran", "alger", "oran"]})
df.describe()
df.groupby("city").mean("age")

Xtr, Xte, ytr, yte = tf.preprocessing.train_test_split(X, y, test_size=0.2, random_state=0)
Xtr = tf.preprocessing.StandardScaler().fit_transform(Xtr)
clf = tf.ml.LogisticRegression(epochs=500).fit(Xtr, ytr)
print(clf.score(Xte, yte), tf.ml.classification_report(yte, clf.predict(Xte)))
```

More in [`examples/`](examples/) and [`docs/API.md`](docs/API.md).

## 🗂️ Project layout

```
tensorf/
├── assets/            brand (logo.svg, banner.svg)
├── benchmarks/        CPU benchmarks vs baselines
├── docs/              API map + guides
├── examples/          quickstart, sql_dataframe, ml_pipeline
├── tensorf/         the library (28 modules, numpy-only core)
│   ├── tensor.py ops.py nn.py autodiff.py   eager tensors + autograd
│   ├── layers.py models.py keras.py         Dense/Conv/RNN/LSTM/GRU/Attention
│   ├── optimizers.py losses.py metrics.py   AdamW/Lion, huber/hinge, F1/AUC
│   ├── callbacks.py schedules.py            EarlyStopping, Cosine/Warmup LR
│   ├── data.py io.py database.py dataframe.py  Dataset, CSV/JSON/NPY, SQLite, DataFrame
│   └── preprocessing.py stats.py ml.py text.py  sklearn-style ML + TF-IDF
├── tests/             pytest suite (15 tests)
└── pyproject.toml
```

## ✅ Development

```bash
pip install -e .[all]
python -m pytest tests -q
python benchmarks/bench.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## 📄 License

MIT © 2026 salim-studio. See [LICENSE](LICENSE).
