# Contributing to tensorfly

Thanks for your interest! tensorfly is a NumPy-only, TensorFlow-compatible ML framework. Keep contributions small, tested, and dependency-free.

## Setup

```bash
git clone https://github.com/salim-studio/tensorfly.git
cd tensorfly
pip install -e .[all]
python -m pytest tests -q
```

## Ground rules

1. **Core stays NumPy-only.** New hard dependencies are rejected; optional integrations (pandas, duckdb, matplotlib) must be lazy imports with a clear error message pointing at the right extra (`pip install "tensorfly[io]"`, `[db]`, `[viz]`).
2. **TF-compatible naming.** Public functions mirror `tensorflow` / `tf.keras` / `sklearn` names where an equivalent exists.
3. **English everywhere.** Code, docstrings, comments, and docs are in English.
4. **Every feature ships with a test** in `tests/` and a runnable snippet (README, `docs/`, or `examples/`).
5. **Benchmarks for perf claims.** If you claim a speedup, add/extend `benchmarks/bench.py` so it is reproducible.

## Pull requests

- One focused change per PR, with tests passing (`python -m pytest tests -q`).
- Update `CHANGELOG.md` under `Unreleased`.
- For new modules, update `docs/API.md` and the layout section of `README.md`.

## Reporting issues

Include: tensorfly version (`tf.__version__`), Python/NumPy versions, minimal reproducer, expected vs actual output.
