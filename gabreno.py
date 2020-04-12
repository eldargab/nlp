from typing import List
from sklearn.base import BaseEstimator, clone as clone_estimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from logging import INFO, WARNING
import numpy as np


def _stdin_print(msg, level=INFO):
    print(msg)


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, costs: List[float], default_group_idx, min_samples=120, threshold_n_folds=10):
        self.costs = costs
        self.default_group_idx = default_group_idx
        self.min_samples = min_samples
        self.threshold_n_folds = threshold_n_folds

        self.estimators_ = None
        self.thresholds_ = None


    @property
    def n_labels(self):
        return len(self.costs)


    def fit(self, X, y, sample_weight=None, log=_stdin_print):
        y = np.asanyarray(y)

        estimators = [None] * self.n_labels
        thresholds = np.zeros(self.n_labels)

        for idx, cost in enumerate(self.costs):
            samples = y == idx
            positives_count = np.sum(samples)
            if positives_count < self.min_samples or len(samples) - positives_count < self.min_samples:
                if idx != self.default_group_idx:
                    log(f"Not enough data for group {idx}. It's messages will be routed to default group.", WARNING)
                continue

            est = LinearSVC()

            thresholds[idx] = self._fit_threshold(est, X, samples, cost, sample_weight)
            est.fit(X, samples, sample_weight)
            estimators[idx] = est

            log(f"Fitted estimator for group {idx}")

        self.estimators_ = estimators
        self.thresholds_ = thresholds


    def _fit_threshold(self, estimator, X, y, error_cost, sample_weight):
        def index_sample_weight(i, w=sample_weight, d=None):
            return w[i] if w is not None else d

        threshold = 0.0

        for train, test in StratifiedKFold(self.threshold_n_folds, shuffle=True).split(X, y):
            e = clone_estimator(estimator)
            e.fit(X[train], y[train], index_sample_weight(train))
            scores = e.decision_function(X[test])
            order = np.flipud(np.argsort(scores))
            positives = y[test][order]
            negatives = ~positives
            economy = np.empty(len(order))
            weights = index_sample_weight(test)
            economy[positives] = index_sample_weight(positives, weights, 1.0) * 1.0
            economy[negatives] = index_sample_weight(negatives, weights, 1.0) * (-error_cost)
            np.cumsum(economy, out=economy)
            idx = np.argmax(economy)
            threshold += scores[order[idx]]

        return threshold / self.threshold_n_folds


    def _compute_scores(self, X):
        scores = np.empty((X.shape[0], self.n_labels), dtype=np.float)

        for i, est in enumerate(self.estimators_):
            if est is None:
                scores[:, i] = 0.0
            else:
                scores[:, i] = est.decision_function(X)

        return scores


    def predict(self, X):
        if self.estimators_ is None:
            raise RuntimeError('Instance not fitted')

        scores = self._compute_scores(X)
        count = scores.shape[0]

        def results():
            for i in range(count):
                guess = self.default_group_idx
                best_score = 0.0
                for j in range(self.n_labels):
                    score = scores[i, j]
                    if score > self.thresholds_[j] and score > best_score:
                        guess = j
                        best_score = score
                yield guess

        return np.fromiter(results(), np.min_scalar_type(self.n_labels), count)


    def score(self, X, y, sample_weight=None):
        return compute_economy(self.costs, self.default_group_idx, y, self.predict(X), sample_weight)


def compute_economy(costs: List[float], default_group_idx: int, y, guesses, sample_weight=None):
    hits = y == guesses
    misses = ~hits
    n_labels = len(costs)

    tp = np.bincount(
        y[hits],
        sample_weight[hits] if sample_weight is not None else None,
        minlength=n_labels
    )[:n_labels]

    fp = np.bincount(
        guesses[misses],
        sample_weight[misses] if sample_weight is not None else None,
        minlength=n_labels
    )[:n_labels]

    tp[default_group_idx] = 0
    fp[default_group_idx] = 0

    economy = np.sum(tp) - np.dot(fp, np.array(costs))
    number_of_samples = len(y) if sample_weight is None else np.sum(sample_weight)

    return economy / number_of_samples

