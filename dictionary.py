from typing import Iterable
import re


TOKEN_SPLIT_REGEX = re.compile(r'[\W_]')


def _tokens(s: str) -> Iterable[str]:
    for tok in TOKEN_SPLIT_REGEX.split(s):
        if tok:
            yield tok.lower()


STANDARD_TOKENS = [
    'UNKNOWN'
]


class Dictionary:
    def __init__(self):
        self._dict = {}
        self._doc_counts = []
        self._doc_count = 0
        self._add_standard_tokens()

    def _add_standard_tokens(self):
        for tok in STANDARD_TOKENS:
            self._add_new_token(tok)

    def reset(self):
        self._dict.clear()
        self._doc_counts.clear()
        self._doc_count = 0
        self._add_standard_tokens()

    def fit(self, doc: str):
        for tok in _tokens(doc):
            idx = self._dict.get(tok, -1)
            if idx < 0:
                self._add_new_token(tok)
            else:
                self._doc_counts[idx] += 1
        self._doc_count += 1

    def _add_new_token(self, tok):
        self._dict[tok] = len(self._dict)
        self._doc_counts.append(1)

    def limit(self, min_docs: int = 5):
        length = 0
        new_counts = []
        deleted = set()

        for tok, idx in self._dict.items():
            if self._doc_counts[idx] < min_docs and idx >= len(STANDARD_TOKENS):
                deleted.add(tok)
            else:
                new_counts.append(self._doc_counts[idx])
                self._dict[tok] = length
                length += 1

        for tok in deleted:
            del self._dict[tok]

        self._doc_counts = new_counts

    def __call__(self, doc: str) -> Iterable[int]:
        for tok in _tokens(doc):
            yield self._dict.get(tok, 0)

    @property
    def size(self):
        return len(self._dict)
