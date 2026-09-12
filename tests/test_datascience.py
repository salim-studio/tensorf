import os
import numpy as np
import tensorf as tf

TMP = os.environ.get("TEMP", "/tmp")


def test_dataframe_basics():
    df = tf.DataFrame({"age": [20, 30, 25], "city": ["oran", "alger", "oran"]})
    assert df.shape == (3, 2)
    assert df["age"].mean() == 25.0
    f = df.filter(df["age"] > 21)
    assert len(f) == 2
    assert df.groupby("city").mean("age")["oran"] == 22.5
    t = df.to_tensor(columns=["age"])
    assert t.shape == (3, 1)


def test_database_roundtrip():
    db = tf.db.Database(":memory:")
    db.execute("CREATE TABLE t (a REAL, b REAL)")
    db.executemany("INSERT INTO t VALUES (?,?)", [(1, 2), (3, 4)])
    assert "t" in db.tables()
    ten = db.read_sql("SELECT * FROM t")
    assert ten.shape == (2, 2)
    df = db.read_dataframe("SELECT * FROM t")
    assert df.shape == (2, 2)
    db.to_sql("t2", df)
    assert "t2" in db.tables()
    db.close()


def test_io_csv_json_npy():
    p1 = os.path.join(TMP, "tfly_x.csv")
    p2 = os.path.join(TMP, "tfly_x.npy")
    p3 = os.path.join(TMP, "tfly_c.json")
    X = tf.constant([[1., 2.], [3., 4.]])
    tf.io.save_csv(p1, X)
    assert tf.io.load_csv(p1).shape == (2, 2)
    tf.io.save_npy(p2, X)
    assert tf.io.load_npy(p2).shape == (2, 2)
    tf.io.save_json(p3, {"lr": 0.01})
    assert tf.io.load_json(p3)["lr"] == 0.01


def test_preprocessing_ml():
    X = np.random.RandomState(0).randn(60, 4).astype(np.float32)
    y = (X.sum(1) > 0).astype(int)
    Xtr, Xte, ytr, yte = tf.preprocessing.train_test_split(X, y, test_size=0.25, random_state=0)
    sc = tf.preprocessing.StandardScaler().fit(Xtr)
    assert sc.transform(Xte).shape == Xte.shape
    clf = tf.ml.LogisticRegression(lr=0.5, epochs=300).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.8
    km = tf.ml.KMeans(k=2, seed=0).fit(X)
    assert km.cluster_centers_.shape == (2, 4)
    assert tf.ml.PCA(n_components=2).fit_transform(X).shape == (60, 2)
    assert tf.ml.DecisionTreeClassifier(max_depth=3).fit(Xtr, ytr).score(Xte, yte) > 0.5


def test_callbacks_schedules_metrics():
    s = tf.schedules.CosineDecay(0.1, 100)
    assert 0 <= s(100) <= 0.1
    m = tf.metrics.F1Score()
    m.update_state([0, 1, 1, 1], [0, 0, 1, 1])
    assert 0 <= m.result() <= 1
    a = tf.metrics.AUC()
    a.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert a.result() > 0.9


def test_new_layers_and_optimizers():
    for opt in ["adamw", "lion", "adadelta"]:
        m = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="relu"),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        X = np.random.randn(40, 4).astype(np.float32)
        Y = (X.sum(1, keepdims=True) > 0).astype(np.float32)
        h = m.fit(X, Y, epochs=1, batch_size=16, verbose=0)
        assert h["loss"][0] > 0
    # LSTM / GRU / Attention forward
    x = tf.constant(np.random.randn(2, 5, 4).astype(np.float32))
    assert tf.keras.layers.LSTM(6)(x).shape == (2, 6)
    assert tf.keras.layers.GRU(6)(x).shape == (2, 6)
    assert tf.keras.layers.MultiHeadAttention(2, 8)(x).shape == (2, 5, 4)
    assert tf.keras.layers.GlobalAveragePooling2D()(tf.constant(np.random.randn(2, 4, 4, 3).astype(np.float32))).shape == (2, 3)


def test_text_utils():
    tok = tf.text.Tokenizer().fit_on_texts(["hello world", "hello tensorf"])
    seq = tok.texts_to_sequences(["hello world"])
    assert seq[0][0] == tok.word_index["hello"]
    M, vocab = tf.text.tfidf_matrix(["hello world", "hello fly"])
    assert M.shape[0] == 2
