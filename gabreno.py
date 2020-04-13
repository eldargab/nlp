from typing import List
from sklearn.base import BaseEstimator, clone as clone_estimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from logging import INFO
import numpy as np

from util import predict_with_threshold, fit_binary_threshold, compute_precision_score


def _stdin_print(msg, level=INFO):
    print(msg)


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, precision: List[float], default_group_idx, threshold_n_folds=10):
        self.precision = precision
        self.default_group_idx = default_group_idx
        self.threshold_n_folds = threshold_n_folds

        self.estimators_ = None
        self.thresholds_ = None


    @property
    def n_labels(self):
        return len(self.precision)


    def fit(self, X, y, log=_stdin_print):
        y = np.asanyarray(y)

        estimators = [None] * self.n_labels
        thresholds = np.zeros(self.n_labels)

        for idx, precision in enumerate(self.precision):
            samples = y == idx
            positives_count = np.sum(samples)
            if positives_count < 100 or len(samples) - positives_count < 100:
                continue

            est = LinearSVC()

            thresholds[idx] = self._fit_threshold(est, X, samples, precision)

            if thresholds[idx] == np.inf:
                log(f"Skipping estimator for group {idx}")
            else:
                est.fit(X, samples)
                estimators[idx] = est
                log(f"Fitted estimator for group {idx}")

        self.estimators_ = estimators
        self.thresholds_ = thresholds


    def _fit_threshold(self, estimator, X, y, precision):
        test_scores = []
        test_y = []

        for train, test in StratifiedKFold(self.threshold_n_folds, shuffle=True).split(X, y):
            e = clone_estimator(estimator)
            e.fit(X[train], y[train])
            scores = e.decision_function(X[test])
            test_scores.append(scores)
            test_y.append(y[test])

        return fit_binary_threshold(np.concatenate(test_scores), np.concatenate(test_y), precision)


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
        return predict_with_threshold(scores, self.thresholds_, self.default_group_idx)


    def score(self, X, y, sample_weight=None):
        return compute_precision_score(self.precision, self.default_group_idx, y, self.predict(X))
