from collections import OrderedDict
from typing import Callable, Iterator, TypeVar, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import binom

T = TypeVar('T')


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


class IterableGenerator:
    def __init__(self, f: Callable[[], Iterator[T]]):
        self.f = f

    def __iter__(self) -> Iterator[T]:
        return self.f()


def iterable_generator(f:  Callable[[], Iterator[T]]) -> Iterable[T]:
    return IterableGenerator(f)


class ArrayBuilder:
    def __init__(self, dtype, initial_size=10, growth=1.25):
        self._growth = growth
        self._cap = initial_size
        self._size = 0
        self._data = np.empty(self._cap, dtype=dtype)

    def append(self, x):
        idx = self._size
        self._grow(self._size + 1)
        self._data[idx] = x

    def append_list(self, x):
        start = self._size
        end = self._size + len(x)
        self._grow(end)
        self._data[start:end] = x

    def _grow(self, size):
        self._size = size
        if size >= self._cap:
            new_cap = round(self._cap * self._growth)
            self._data.resize((new_cap,), refcheck=False)
            self._cap = new_cap

    def end(self):
        data = self._data
        data.resize((self._size,), refcheck=False)
        self._data = np.empty(0, dtype=data.dtype)
        self._size = 0
        self._cap = 0
        return data


class RaggedPaddedBatches:
    @staticmethod
    def from_list(data, batch_size, padding_value=0, dtype=None, size_dtype=np.int):
        batches_count = len(data) // batch_size
        offsets = np.empty(batches_count + 1, dtype=size_dtype)
        offsets[0] = 0
        data_size = 0
        for bi in range(0, batches_count):
            offsets[bi] = data_size
            start = bi * batch_size
            end = start + batch_size
            batch_item_size = max((len(data[i]) for i in range(start, end)), default=0)
            data_size += batch_item_size * batch_size
            offsets[bi + 1] = data_size

        data_array = np.empty(data_size, dtype=dtype if dtype else np.array(data[0]).dtype)
        offset = 0
        for i in range(0, batches_count * batch_size):
            data_item = data[i]
            bi = i // batch_size
            size = (offsets[bi + 1] - offsets[bi]) // batch_size
            end = offset + size
            item_end = offset+len(data_item)
            data_array[offset:item_end] = data_item
            if end > item_end:
                data_array[item_end:end] = padding_value
            offset = end

        return RaggedPaddedBatches(data_array, offsets, batch_size)

    def __init__(self, data, offsets, batch_size):
        self._data = data
        self._offsets = offsets
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batches_count(self):
        return len(self._offsets) - 1

    @property
    def size(self):
        return self.batches_count * self.batch_size

    def shuffled_tensor_batches(self, dtype=None, y=None):
        import torch

        rng = np.random.default_rng(0)
        batches = rng.permutation(self.batches_count)
        tensor = torch.tensor(self._data, dtype=dtype)

        @iterable_generator
        def iterable():
            for batch in batches:
                start = self._offsets[batch]
                end = self._offsets[batch + 1]
                x = tensor[start:end].view(self.batch_size, -1)
                if y is None:
                    yield x
                else:
                    ys = batch * self.batch_size
                    ye = ys + self.batch_size
                    yield x, y[ys:ye]

        return iterable

    def tensor_batches(self, dtype=None):
        import torch

        tensor = torch.tensor(self._data, dtype=dtype)

        @iterable_generator
        def iterable():
            for batch in range(0, self.batches_count):
                start = self._offsets[batch]
                end = self._offsets[batch + 1]
                yield tensor[start:end].view(self.batch_size, -1)

        return iterable


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


def fit_binary_threshold(score: np.ndarray, y: np.ndarray, precision: np.float) -> float:
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

    if binom.cdf(fp_count, n, 1-precision) > 0.3:
        return np.inf

    if idx + 1 == len(y):
        return score[order[idx]]
    else:
        return np.mean([score[order[idx + 1]], score[order[idx]]])


def fit_threshold_vector(scores: np.ndarray, y: np.ndarray, precision: np.ndarray) -> np.ndarray:
    thresholds = np.zeros_like(precision)
    for i, p in enumerate(precision):
        i_scores = scores[:, i]
        thresholds[i] = fit_binary_threshold(i_scores, y == i, p)
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


class NNClassifierTraining:
    def __init__(self, x, y, precision, default_group_idx):
        self.x = x
        self.y = y
        self.precision = precision
        self.default_group_idx = default_group_idx

    def new_model(self):
        raise NotImplementedError()

    def train_epoch(self, model, x, y):
        raise NotImplementedError()

    def shuffle_xy(self, seed=None):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(self.y))
        x = self.x[permutation]
        y = self.y[permutation]
        return x, y

    def run(self):
        threshold, n_epochs = self._fit_threshold_and_epochs()
        model = self._fit_n_epochs_model(n_epochs)
        return ProbThresholdModel(model, threshold, self.default_group_idx)

    def _fit_n_epochs_model(self, n_epochs, model_idx=1):
        x, y = self.shuffle_xy()
        model = self.new_model()

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(model, x, y)
            print(f'model: {model_idx}, epoch: {epoch}, loss: {loss}')

        return model

    def _fit_threshold_and_epochs(self):
        from sklearn.model_selection import StratifiedKFold

        thresholds = []
        epochs = []

        k_folds = StratifiedKFold(10, shuffle=True).split(self.x, self.y)

        for k_fold_idx, (train_set, test_set) in enumerate(k_folds):
            th, epoch = self._fit_k_fold(k_fold_idx, train_set, test_set)
            thresholds.append(th)
            epochs.append(epoch)

        return np.mean(thresholds, axis=0), round(np.mean(epochs))


    def _fit_k_fold(self, k_fold_idx, train_set, test_set):
        model = self.new_model()

        x_train, y_train = self.x[train_set], self.y[train_set]
        x_test, y_test = self.x[test_set], self.y[test_set]

        prev_score = None
        prev_threshold = None

        for epoch in range(1, 50):
            loss = self.train_epoch(model, x_train, y_train)

            prob = model.predict_prob(x_test)

            threshold = fit_threshold_vector(prob, y_test, self.precision)

            score = compute_precision_score(
                self.precision,
                self.default_group_idx,
                y_test,
                predict_with_threshold(prob, threshold, self.default_group_idx)
            )

            print(f'k-fold: {k_fold_idx}, epoch: {epoch}, loss: {round(loss, 2)}, v-score: {round(score, 2)}')

            if prev_score is not None and prev_score < score:
                return prev_threshold, epoch - 1
            else:
                prev_score = score
                prev_threshold = threshold

        return prev_threshold, epoch

