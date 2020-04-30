from abc import ABC
from collections import OrderedDict
from typing import Callable, Iterator, TypeVar, Iterable, List, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import binom
import dask



def frequencies(size_series: pd.Series):
    length = size_series.sum()
    size = size_series.sort_values(ascending=False)
    freq = size / length
    return pd.DataFrame.from_dict(OrderedDict([
        ('Size', size),
        ('Freq', freq),
        ('FreqCumSum', freq.cumsum()),
        ('SizeRevCumSum', size[::-1].cumsum()[::-1])
    ]))


def groups_summary(df: pd.DataFrame):
    sizes = df.groupby('Group').size()
    sizes = sizes[sizes > 0]
    return frequencies(sizes)


def group_size_by_week(df: pd.DataFrame, group: str):
    return df[df.Group == group].resample('W-MON', on='Date').size()


def f_score(p, r, beta=1):
    return p * r * (1 + beta * beta) / (p * beta * beta + r)


def describe_results(group: pd.Series, guess, f_score_beta=1):
    hits = group == guess
    misses = ~hits

    stats = pd.DataFrame.from_dict(OrderedDict([
        ('Group', group),
        ('Guess', guess),
        ('Hit', hits),
        ('Miss', misses)
    ]))

    s = stats.groupby(['Group']).agg(
        Size=pd.NamedAgg('Hit', 'size'),
        Tp=pd.NamedAgg('Hit', 'sum')
    )

    fp = stats.groupby(['Guess']).agg(Fp=pd.NamedAgg('Miss', 'sum'))

    s = s.join(fp, how='outer')
    s.drop('other', inplace=True)
    s.set_index(s.index.add_categories('Overall'), inplace=True)

    overall_row = pd.DataFrame({
        'Size': [len(group)],
        'Tp': [s.Tp.sum()],
        'Fp': [s.Fp.sum()]
    }, index=['Overall'])

    s = s.append(overall_row)

    s['Precision'] = (s.Tp / (s.Tp + s.Fp)).map(lambda x: round(100 * x, ndigits=1))
    s['Recall'] = (s.Tp / s.Size).map(lambda x: round(100 * x, ndigits=1))
    s[f'F{f_score_beta}-score'] = f_score(s.Precision, s.Recall, beta=f_score_beta).map(lambda x: round(x, ndigits=1))
    s.sort_values(by='Size', ascending=False, inplace=True)

    s.fillna(0, inplace=True)
    s['Tp'] = s.Tp.astype(int)
    s['Fp'] = s.Fp.astype(int)
    return s


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, x) -> np.ndarray:
        y = np.array([m.predict(x) for m in self.models], order='F')
        n_labels, n_samples = y.shape
        out = np.empty(n_samples, dtype=np.int)
        for i in range(0, n_samples):
            votes = np.bincount(y[:, i], minlength=n_labels)
            out[i] = votes.argmax()
        return out

    def predict_prob(self, x) -> np.ndarray:
        return np.mean([m.predict_prob(x) for m in self.models], axis=0)


class ProbThresholdModel:
    def __init__(self, model, threshold, default_group_idx):
        self.model = model
        self.threshold = threshold
        self.default_group_idx = default_group_idx

    def predict(self, x) -> np.ndarray:
        prob = self.model.predict_prob(x)
        return predict_with_threshold(prob, self.threshold, self.default_group_idx)

    def predict_prob(self, x) -> np.ndarray:
        return self.model.predict_prob(x)


def predict_with_threshold(scores: np.ndarray, threshold: np.ndarray, default_group_idx):
    y = scores.argmax(axis=1)
    diff = scores - threshold
    y[diff[np.arange(0, len(y)), y] <= 0] = default_group_idx
    return y


def fit_binary_threshold(score: np.ndarray, y: np.ndarray, precision: np.float, alpha=0.3) -> float:
    fp_cost = np.divide(precision, np.subtract(1, precision))
    order = np.flipud(np.argsort(score))
    cum_score = np.empty(len(y))
    positives = y[order]
    negatives = ~positives
    cum_score[positives] = 1.0
    cum_score[negatives] = -1 * fp_cost
    np.cumsum(cum_score, out=cum_score)
    idx = np.argmax(cum_score)

    if cum_score[idx] <= 0:
        return np.inf

    n = idx + 1
    fp_count = np.sum(negatives[0:idx+1])

    if binom.cdf(fp_count, n, 1-precision) > alpha:
        return np.inf

    if idx + 1 == len(y):
        return score[order[idx]]
    else:
        return np.mean([score[order[idx + 1]], score[order[idx]]])


def fit_threshold_vector(scores: np.ndarray, y: np.ndarray, precision: np.ndarray, alpha=0.3) -> np.ndarray:
    thresholds = np.zeros_like(precision)
    for i, p in enumerate(precision):
        i_scores = scores[:, i]
        thresholds[i] = fit_binary_threshold(i_scores, y == i, p, alpha)
    return thresholds


def compute_precision_score(precision, default_group_idx: int, y, guesses):
    costs = np.divide(precision, np.subtract(1, precision))
    hits = y == guesses
    misses = ~hits
    n_labels = len(costs)

    tp = np.bincount(
        y[hits],
        minlength=n_labels
    )[:n_labels]

    fp = np.bincount(
        guesses[misses],
        minlength=n_labels
    )[:n_labels]

    tp[default_group_idx] = 0
    fp[default_group_idx] = 0

    score = np.sum(tp) - np.dot(fp, costs)
    return score / len(y)
