import numpy as np
import math


def softmax(x: np.ndarray) -> np.ndarray:
    """
    :param x: 1-D vector of softmax weights
    :return: 1-D vector of probabilities
    """
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s


class FastText:
    def __init__(self, dict_size: int, dict_dim: int, n_labels: int,):
        self.embedding = np.random.default_rng().uniform(-1.0 / dict_dim, 1.0 / dict_dim, (dict_size, dict_dim))
        self.embedding[0] = 0
        self.output_weights = np.zeros((n_labels, dict_dim))

    def forward(self, sentence: np.ndarray) -> np.ndarray:
        x = np.mean(self.embedding[sentence], axis=0)
        x = np.matmul(self.output_weights, x)
        return softmax(x)

    def backward(self, sentence: np.ndarray, y: int, lr) -> float:
        x = np.mean(self.embedding[sentence], axis=0)
        p = softmax(np.matmul(self.output_weights, x))
        alpha = p * (-1.0 * lr)
        alpha[y] = (1 - p[y]) * lr
        self.embedding[sentence] += np.matmul(alpha, self.output_weights) / len(sentence)
        self.output_weights += np.outer(alpha, x)
        return -math.log(p[y])

