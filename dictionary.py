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
        self.dict = {}
        self.doc_counts = []
        self.doc_count = 0
        self._add_standard_tokens()

    def _add_standard_tokens(self):
        for tok in STANDARD_TOKENS:
            self._add_new_token(tok)

    def reset(self):
        self.dict.clear()
        self.doc_counts.clear()
        self.doc_count = 0
        self._add_standard_tokens()

    def fit(self, doc: str):
        for tok in _tokens(doc):
            idx = self.dict.get(tok, -1)
            if idx < 0:
                self._add_new_token(tok)
            else:
                self.doc_counts[idx] += 1
        self.doc_count += 1

    def _add_new_token(self, tok):
        self.dict[tok] = len(self.dict)
        self.doc_counts.append(1)

    def limit(self, min_docs: int = 5):
        length = 0
        new_counts = []
        deleted = set()

        for tok, idx in self.dict.items():
            if self.doc_counts[idx] < min_docs and idx >= len(STANDARD_TOKENS):
                deleted.add(tok)
            else:
                new_counts.append(self.doc_counts[idx])
                self.dict[tok] = length
                length += 1

        for tok in deleted:
            del self.dict[tok]

        self.doc_counts = new_counts

    def __call__(self, doc: str) -> Iterable[int]:
        for tok in _tokens(doc):
            yield self.dict.get(tok, 0)
