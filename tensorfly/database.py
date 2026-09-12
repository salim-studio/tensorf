"""tensorfly.database — database tasks (SQLite by default + optional duckdb/SQLAlchemy).

    import tensorfly as tf

    db = tf.db.Database("app.db")
    db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT, age REAL)")
    db.executemany("INSERT INTO users VALUES (?,?,?)", [(1,"salim",30),(2,"amina",25)])
    t = db.read_sql("SELECT * FROM users WHERE age > 20")   # -> Tensor (numeric cols)
    df = db.read_dataframe("SELECT * FROM users")            # -> DataFrame (mixed types)
    db.to_sql("backup_users", df)                            # DataFrame/Tensor -> table

Designed like pandas.read_sql + sqlalchemy-lite: any DB-API connection works.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Sequence

import numpy as np


def _is_pandas_df(x) -> bool:
    return type(x).__name__ == "DataFrame" and hasattr(x, "to_sql")


class Database:
    """Lightweight wrapper around sqlite3 (or any DB-API connection)."""

    def __init__(self, path=":memory:", connection=None, timeout=30.0):
        if connection is not None:
            self.conn = connection
            self.path = getattr(connection, "database", ":external:")
        else:
            self.path = path
            self.conn = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        self._closed = False

    # -- core --
    def execute(self, sql, params=None):
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        self.conn.commit()
        return cur

    def executemany(self, sql, seq):
        cur = self.conn.cursor()
        cur.executemany(sql, seq)
        self.conn.commit()
        return cur

    def executescript(self, script):
        cur = self.conn.cursor()
        cur.executescript(script)
        self.conn.commit()
        return cur

    def query(self, sql, params=None) -> list:
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        return cur.fetchall()

    # -- schema helpers --
    def tables(self) -> list:
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        try:
            return [r["name"] for r in rows]
        except Exception:
            return [tuple(r)[0] for r in rows]

    def schema(self, table) -> list:
        return self.query(f"PRAGMA table_info({table})")

    def drop_table(self, table, if_exists=True):
        self.execute(f"DROP TABLE {'IF EXISTS' if if_exists else ''} {table}".replace("  ", " "))

    # -- read helpers --
    def read_sql(self, sql, params=None, dtype=np.float32):
        """SELECT -> Tensor (numeric columns only; text becomes NaN)."""
        from .tensor import convert_to_tensor
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        if not rows:
            return convert_to_tensor(np.empty((0, 0), dtype=np.float32))
        arr = []
        for r in rows:
            vals = []
            for v in tuple(r):
                try:
                    vals.append(float(v) if v is not None else np.nan)
                except Exception:
                    vals.append(np.nan)
            arr.append(vals)
        return convert_to_tensor(np.asarray(arr, dtype=np.float32), dtype=dtype)

    def read_dataframe(self, sql, params=None):
        """SELECT -> DataFrame (keeps mixed types and column names)."""
        from .dataframe import DataFrame
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        data = {c: [] for c in cols}
        for r in rows:
            for c, v in zip(cols, tuple(r)):
                data[c].append(v)
        return DataFrame(data)

    def read_pandas(self, sql, params=None):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("read_pandas needs pandas: pip install 'tensorfly[io]'") from e
        return pd.read_sql(sql, self.conn, params=params)

    # -- write helpers --
    def to_sql(self, table, data, if_exists="replace", index=False):
        """DataFrame/Tensor/dict/list -> SQL table."""
        from .dataframe import DataFrame
        from .tensor import Tensor
        if isinstance(data, DataFrame):
            cols = data.columns
            _, mat = data._matrix(object)
        elif isinstance(data, Tensor):
            a = data._data
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            cols = [f"c{i}" for i in range(a.shape[1])]
            mat = a.tolist()
        elif isinstance(data, dict):
            cols = list(data.keys())
            mat = list(map(list, zip(*[list(v) for v in data.values()]))) if cols else []
        else:
            a = np.asanyarray(data)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            cols = [f"c{i}" for i in range(a.shape[1])]
            mat = a.tolist()
        if if_exists == "replace":
            self.execute(f"DROP TABLE IF EXISTS {table}")
        elif if_exists == "fail" and table in self.tables():
            raise ValueError(f"table '{table}' exists")
        # infer types from first non-None value
        coltypes = []
        for j, c in enumerate(cols):
            t = "TEXT"
            for row in mat:
                v = row[j] if j < len(row) else None
                if v is None:
                    continue
                if isinstance(v, (int, np.integer)):
                    t = "INTEGER"
                elif isinstance(v, (float, np.floating)):
                    t = "REAL"
                break
            coltypes.append(f'"{c}" {t}')
        if if_exists != "append" or table not in self.tables():
            self.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(coltypes)})')
        ph = ", ".join(["?"] * len(cols))
        names = ", ".join([f'"{c}"' for c in cols])
        self.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({ph})', mat)
        return table

    def close(self):
        try:
            self.conn.commit()
            if self.path != ":external:":
                self.conn.close()
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return f"tensorfly.Database(path={self.path!r}, tables={self.tables() if not self._closed else '?'})"


# -- functional API (like pandas.read_sql) --
def connect(path=":memory:", **kw) -> Database:
    return Database(path, **kw)


def read_sql(sql, con, params=None, dtype=np.float32):
    db = con if isinstance(con, Database) else Database(con)
    out = db.read_sql(sql, params, dtype)
    if not isinstance(con, Database):
        db.close()
    return out


def read_dataframe(sql, con, params=None):
    db = con if isinstance(con, Database) else Database(con)
    out = db.read_dataframe(sql, params)
    if not isinstance(con, Database):
        db.close()
    return out


def to_sql(table, data, con, if_exists="replace"):
    db = con if isinstance(con, Database) else Database(con)
    out = db.to_sql(table, data, if_exists)
    if not isinstance(con, Database):
        db.close()
    return out


def read_duckdb(sql, path=":memory:"):
    """DuckDB query (optional, faster for large analytics) -> DataFrame."""
    try:
        import duckdb
    except ImportError as e:
        raise ImportError("read_duckdb needs duckdb: pip install 'tensorfly[db]'") from e
    from .dataframe import DataFrame
    rel = duckdb.connect(path).execute(sql).fetchdf()
    return DataFrame.from_pandas(rel)


__all__ = ["Database", "connect", "read_sql", "read_dataframe", "to_sql", "read_duckdb"]
