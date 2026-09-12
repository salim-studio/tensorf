"""Classical ML pipeline: scale -> logistic regression -> report (+ KMeans/PCA)."""
import numpy as np

import tensorfly as tf

tf.utils.seed_everything(0)
X = np.random.default_rng(0).standard_normal((300, 4)).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(int)

Xtr, Xte, ytr, yte = tf.preprocessing.train_test_split(X, y, test_size=0.25, random_state=0)
Xtr = tf.preprocessing.StandardScaler().fit_transform(Xtr)

clf = tf.ml.LogisticRegression(lr=0.5, epochs=300).fit(Xtr, ytr)
print("accuracy:", clf.score(Xte, yte))
print(tf.ml.classification_report(yte, clf.predict(Xte)))

print("kmeans centers:", tf.ml.KMeans(k=2, seed=0).fit(X).cluster_centers_.shape)
print("pca:", tf.ml.PCA(n_components=2).fit_transform(X).shape)
