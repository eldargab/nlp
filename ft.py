import numpy as np
import math
import itertools


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


def train(sentence_seq, y: np.ndarray, dict_size: int, n_labels: int, model_idx: int = 0) -> FastText:
    model = FastText(dict_size=dict_size, dict_dim=100, n_labels=n_labels)
    n_samples = len(y)
    permutation = np.random.default_rng().permutation(n_samples)

    vset_size = n_samples // 5
    vset_idx = permutation[0:vset_size]
    vset_x = sentence_seq[vset_idx]
    vset_y = y[vset_idx]

    prev_vset_loss = None
    prev_checkpoint = None

    for epoch in range(0, 30):
        loss = 0.0

        for i in range(vset_size, n_samples):
            loss += model.backward(sentence_seq[permutation[i]], y[permutation[i]], lr=0.2)

        loss = loss / n_samples
        vset_loss = model.loss(vset_x, vset_y)
        print(f'model: {model_idx}, epoch: {epoch}, loss: {round(loss, 2)}, vset loss: {round(vset_loss, 2)}')
        if prev_vset_loss is not None and prev_vset_loss < vset_loss:
            model.restore_checkpoint(prev_checkpoint)
            break
        else:
            prev_vset_loss = vset_loss
            prev_checkpoint = model.save_checkpoint()

    return model
