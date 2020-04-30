from ft import softmax
import numpy as np
import math


class FastText2:
    def __init__(self, dict_size: int, dict_dim: int, hidden_dim: int, n_labels: int,):
        rng = np.random.default_rng()
        self.embedding = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (dict_size, dict_dim))
        self.embedding[0] = 0
        self.hidden_weights = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (hidden_dim, dict_dim))
        self.hidden_bias = rng.uniform(-1.0 / hidden_dim, 1.0 / hidden_dim, hidden_dim)
        self.output_weights = rng.uniform(-1.0 / hidden_dim, 1.0 / hidden_dim, (n_labels, hidden_dim))


    def forward(self, sentence: np.ndarray) -> np.ndarray:
        x = np.mean(self.embedding[sentence], axis=0)
        x = np.matmul(self.hidden_weights, x) + self.hidden_bias
        x[x < 0] = 0.0
        x = np.matmul(self.output_weights, x)
        return softmax(x)


    def backward(self, sentence: np.ndarray, y: int, lr) -> float:
        x = np.mean(self.embedding[sentence], axis=0)
        h = np.matmul(self.hidden_weights, x) + self.hidden_bias
        h_zeros = h < 0
        h[h_zeros] = 0.0
        o = np.matmul(self.output_weights, h)
        p = softmax(o)

        d_o = p * (-1.0 * lr)
        d_o[y] = (1 - p[y]) * lr

        d_h = np.matmul(d_o, self.output_weights)
        d_h[h_zeros] = 0.0

        d_x = np.matmul(d_h, self.hidden_weights)

        self.output_weights += np.outer(d_o, h)
        self.hidden_weights += np.outer(d_h, x)
        self.hidden_bias += d_h
        self.embedding[sentence] += d_x / len(sentence)

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
        return self.embedding.copy(), self.hidden_weights.copy(), self.hidden_bias.copy(), self.output_weights.copy()

    def restore_checkpoint(self, checkpoint):
        self.embedding = checkpoint[0].copy()
        self.hidden_weights = checkpoint[1].copy()
        self.hidden_bias = checkpoint[2].copy()
        self.output_weights = checkpoint[3].copy()
