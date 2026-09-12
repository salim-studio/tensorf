"""SQL -> DataFrame -> Tensor -> model (the tensorf data-stack loop)."""
import tensorf as tf

with tf.db.Database(":memory:") as db:
    db.execute("CREATE TABLE users (id INTEGER, name TEXT, age REAL, income REAL)")
    db.executemany("INSERT INTO users VALUES (?,?,?,?)", [
        (1, "salim", 30, 4200.0),
        (2, "amina", 25, 3100.0),
        (3, "yousef", 41, 5800.0),
        (4, "lina", 29, 3600.0),
    ])
    print("tables:", db.tables())

    df = db.read_dataframe("SELECT * FROM users WHERE age >= 25")
    df.head()
    print(df.describe())
    print("mean income:", df["income"].mean())

    X = df.to_tensor(columns=["age", "income"])
    print("training tensor:", X.shape)
