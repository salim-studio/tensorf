"""tensorfly.text — text encoding for NLP (Tokenizer + ngrams + light tfidf)."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text, lower=True) -> list:
    if lower:
        text = text.lower()
    return _WORD.findall(text)


def ngrams(tokens, n=2) -> list:
    return [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


class Tokenizer:
    """Like the keras Tokenizer: fit_on_texts + texts_to_sequences + pad."""
    def __init__(self, num_words=None, oov_token="<OOV>", lower=True):
        self.num_words = num_words
        self.oov_token = oov_token
        self.lower = lower
        self.word_index: dict[str, int] = {}
        self.index_word: dict[int, str] = {}

    def fit_on_texts(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(tokenize(t, self.lower))
        vocab = [w for w, _ in cnt.most_common(self.num_words - 1 if self.num_words else None)]
        self.word_index = {}
        if self.oov_token:
            self.word_index[self.oov_token] = 1
        for i, w in enumerate(vocab, start=len(self.word_index) + 1):
            self.word_index[w] = i
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    def texts_to_sequences(self, texts):
        oov = self.word_index.get(self.oov_token, 1)
        out = []
        for t in texts:
            out.append([self.word_index.get(w, oov) for w in tokenize(t, self.lower)])
        return out

    def sequences_to_texts(self, seqs):
        return [" ".join(self.index_word.get(i, self.oov_token or "") for i in s) for s in seqs]


def tfidf_matrix(texts, max_features=None, lower=True):
    """Dense TF-IDF matrix (for small/medium data)."""
    docs = [tokenize(t, lower) for t in texts]
    df = Counter()
    for d in docs:
        df.update(set(d))
    vocab = [w for w, _ in Counter({w: df[w] for w in df}).most_common(max_features)]
    idx = {w: i for i, w in enumerate(vocab)}
    N = len(docs)
    M = np.zeros((N, len(vocab)), np.float32)
    for di, d in enumerate(docs):
        c = Counter(d)
        L = len(d) or 1
        for w, f in c.items():
            if w in idx:
                M[di, idx[w]] = (f / L) * np.log((1 + N) / (1 + df[w]) + 1)
    return M, vocab


def bag_of_words(texts, vocab=None, lower=True):
    docs = [tokenize(t, lower) for t in texts]
    if vocab is None:
        vocab = sorted({w for d in docs for w in d})
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)), np.float32)
    for di, d in enumerate(docs):
        for w in d:
            if w in idx:
                M[di, idx[w]] += 1
    return M, vocab


__all__ = ["tokenize", "ngrams", "Tokenizer", "tfidf_matrix", "bag_of_words"]
