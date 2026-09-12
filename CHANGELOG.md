# Changelog

All notable changes to tensorfly are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2026-09-12
### Added
- Full data stack: `tf.io` (CSV/JSON/NPY/NPZ/TXT/Parquet), `tf.db` (SQLite + DuckDB/SQLAlchemy bridges, `read_sql`/`read_dataframe`/`to_sql`), `tf.DataFrame` (head/describe/groupby/merge/corr), `tf.stats`, `tf.viz`.
- Classical ML (`tf.ml`): Linear/Ridge/Logistic regression, KNN, GaussianNB, KMeans, PCA, CART decision trees, accuracy/R²/confusion-matrix/classification-report helpers.
- NLP utilities (`tf.text`): Tokenizer, n-grams, bag-of-words, TF-IDF.
- Deep learning: LSTM, GRU, MultiHeadAttention, GlobalAverage/MaxPooling2D, Reshape, Activation, Add, Concatenate layers; AdamW, Lion, Adadelta optimizers; huber/hinge losses; Precision/Recall/F1/AUC metrics; callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger); LR schedules (Cosine, WarmupCosine, Step, Exponential, Polynomial).
- Training upgrades: `validation_split`, per-epoch metrics + `val_*` history, `evaluate()` returns loss + metrics dict, `save()` for full models.
- Standalone repository with English docs, brand assets, CI, and examples.

## [0.1.0] - 2026-09-10
### Added
- Initial release: eager `Tensor`/`Variable` with autograd (`GradientTape`), TF-style ops, `nn` (activations, conv2d, pooling, fused dense), `keras` Sequential/Model with Dense/Conv2D/RNN layers, SGD/Adam/RMSprop/Adagrad, `data.Dataset` with prefetch, `tf.function`, `random`, multithreaded CPU engine.
