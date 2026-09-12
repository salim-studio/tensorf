# tensorfly API map

One page to find anything. All imports work as `import tensorfly as tf`.

## Core tensors (TF-compatible)

| Module | Contents |
|---|---|
| `tf.tensor` | `Tensor`, `Variable`, `constant`, `zeros/ones/eye/fill/range/linspace/one_hot/cast`, math (`add…pow`, `exp/log/sqrt/square`), `matmul`, `reduce_sum/mean`, `reshape`, `transpose` |
| `tf.ops` | shape (`expand_dims/squeeze/concat/stack/split/tile/pad/gather/where`), `clip_by_value`, `maximum/minimum`, `tensordot/einsum/norm`, reductions (`reduce_max/min/prod`, `argmax/argmin`, `sort/argsort`), comparisons, `linalg` (inv/det/solve/eig/svd/qr) |
| `tf.nn` | `relu/sigmoid/tanh/softmax/log_softmax/gelu/silu/elu/leaky_relu/softplus/dropout`, `conv2d`, `max_pool/avg_pool`, `dense_fused`, `sparse_softmax_cross_entropy` |
| `tf.autodiff` | `GradientTape(persistent=False)` → `.gradient(target, sources)`, `.watch()`, `.jacobian()` |
| `tf.function` | `@tf.function` / `tf.jit` decorator (trace once, fast path after) |
| `tf.random` | `normal/uniform/truncated_normal/randint/shuffle`, `set_seed` |
| `tf.dtypes` | `float32/int32/…` aliases mirroring `tf.*` |
| `tf.parallel` | `set_workers(n)`, `get_workers()` — thread-pool engine tuning |

## Deep learning (Keras-compatible)

| Module | Contents |
|---|---|
| `tf.keras.layers` | `Dense`, `Conv2D`, `MaxPooling2D/AveragePooling2D`, `GlobalAveragePooling2D/GlobalMaxPooling2D`, `Flatten/Reshape`, `Dropout`, `Embedding`, `BatchNormalization/LayerNormalization`, `SimpleRNN/LSTM/GRU`, `MultiHeadAttention`, `Activation/Add/Concatenate`, `Input` |
| `tf.keras.models` | `Sequential` / `Model`: `compile(optimizer, loss, metrics)`, `fit(epochs, batch_size, validation_split/data, callbacks)`, `evaluate`, `predict`, `save_weights/load_weights`, `save`, `summary` |
| `tf.optimizers` | `SGD(momentum/nesterov)`, `Adam` (fused), `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `Lion`; any LR may be a `tf.schedules` callable |
| `tf.losses` | `mse/mae`, `binary/categorical/sparse_categorical_crossentropy`, `huber`, `hinge` |
| `tf.metrics` | `Accuracy`, `Mean/MeanSquaredError/MeanAbsoluteError`, `Precision/Recall/F1Score/AUC` |
| `tf.callbacks` | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger`, `History`, `LambdaCallback` |
| `tf.schedules` | `Constant/ExponentialDecay/StepDecay/CosineDecay/PolynomialDecay/WarmupCosine` |

## Data stack

| Module | Contents |
|---|---|
| `tf.data` | `Dataset.from_tensor_slices/range` → `.map/.batch/.shuffle/.take/.repeat/.prefetch` |
| `tf.io` | `load/save_csv`, `load/save_txt`, `load/save_npy`, `load/save_npz`, `load/save_json`, `read/write_parquet` (needs `[io]`), `load_image` (needs pillow) |
| `tf.db` | `Database(path)` — `execute/query`, `tables/schema`, `read_sql→Tensor`, `read_dataframe→DataFrame`, `to_sql`, `read_pandas` (needs `[io]`), `read_duckdb` (needs `[db]`) |
| `tf.dataframe` | `DataFrame`: `head/tail/describe/dtypes/isna/dropna/fillna/sort_values/groupby/value_counts/corr/merge/to_numpy/to_tensor/to_pandas/to_csv/to_sql` |
| `tf.preprocessing` | `Standard/MinMax/RobustScaler`, `Normalizer`, `Label/OneHotEncoder`, `SimpleImputer`, `train_test_split`, `to_categorical`, `pad_sequences`, `normalize`, `polynomial_features` |
| `tf.stats` | `mean/std/var/median/percentile/corrcoef/cov/skew/kurtosis/zscore/histogram/describe/correlation_matrix` |
| `tf.ml` | `LinearRegression/Ridge/LogisticRegression/KNNClassifier/KNNRegressor/GaussianNB/KMeans/PCA/DecisionTreeClassifier` + `accuracy_score/mean_squared_error/mean_absolute_error/r2_score/confusion_matrix/classification_report` |
| `tf.text` | `tokenize/ngrams`, `Tokenizer`, `tfidf_matrix`, `bag_of_words` |
| `tf.utils` | `seed_everything`, `to_categorical/to_tensor/to_numpy`, `save/load`, `count_params` |
| `tf.viz` | `plot_history/plot_confusion_matrix/scatter` (needs `[viz]`) |
