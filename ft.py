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
        rng = np.random.default_rng()
        self.embedding = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (dict_size, dict_dim))
        self.embedding[0] = 0
        self.output_weights = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (n_labels, dict_dim))

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

    def predict_prob(self, sentence_seq):
        out = np.empty((len(sentence_seq), self.output_weights.shape[0]))
        for i, s in enumerate(sentence_seq):
            out[i] = self.forward(s)
        return out

    def predict(self, sentence_seq):
        prob = self.predict_prob(sentence_seq)
        return prob.argmax(axis=1)

    def loss(self, sentence_seq, y: np.ndarray):
        prob = self.predict_prob(sentence_seq)
        row_range = np.arange(0, len(y))
        return -np.sum(np.log(prob[row_range, y])) / len(y)

    def save_checkpoint(self):
        return self.embedding.copy(), self.output_weights.copy()

    def restore_checkpoint(self, checkpoint):
        self.embedding = checkpoint[0].copy()
        self.output_weights = checkpoint[1].copy()
