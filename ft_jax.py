import numpy as np
import jax.numpy as jnp
import jax
import util


def softmax(x):
    """
    :param x: 1-D vector of softmax weights
    :return: 1-D vector of probabilities
    """
    x = x - jnp.max(x)
    e = jnp.exp(x)
    s = jnp.sum(e)
    return e / s


@jax.jit
def forward(params):
    sentence_embedding, output_weights = params
    x = jnp.matmul(output_weights, sentence_embedding)
    return softmax(x)


def loss_value(params, y: int):
    prob = forward(params)
    return -jnp.log(prob[y])


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_value))


class FastText:
    def __init__(self, dict_size: int, dict_dim: int, n_labels: int,):
        rng = np.random.default_rng()
        self.embedding = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (dict_size, dict_dim))
        self.embedding[0] = 0
        self.output_weights = rng.uniform(-1.0 / dict_dim, 1.0 / dict_dim, (n_labels, dict_dim))

    def forward(self, sentence):
        x = np.mean(self.embedding[sentence], axis=0)
        return np.array(forward((x, self.output_weights)), copy=False)

    def backward(self, sentence: np.ndarray, y: int, lr) -> float:
        x = np.mean(self.embedding[sentence], axis=0)
        loss, (grad_x, grad_w) = loss_value_and_grad((x, self.output_weights), y)
        self.embedding[sentence] -= np.array(grad_x, copy=False) * lr / len(sentence)
        self.output_weights -= np.array(grad_w, copy=False) * lr
        return loss

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


class Training:
    def __init__(self, dict_size, dict_dim, n_labels, **kwargs):
        self.dict_size = dict_size
        self.dict_dim = dict_dim
        self.n_labels = n_labels
        super().__init__(**kwargs)

    def new_model(self):
        return FastText(self.dict_size, self.dict_dim, n_labels=self.n_labels)

    def train_epoch(self, model: FastText, x, y):
        loss = 0.0
        for i in range(0, len(y)):
            loss += model.backward(x[i], y[i], lr=0.2)
        return loss / len(y)


class ClassifierTraining(Training, util.NNClassifierTraining):
    def __init__(self, x, y, precision, default_group_idx, dict_size, dict_dim=100):
        args = dict(
            x=x,
            y=y,
            precision=precision,
            default_group_idx=default_group_idx,
            dict_size=dict_size,
            dict_dim=dict_dim,
            n_labels=len(precision)
        )
        super().__init__(**args)


class RegressionTraining(Training, util.NNTraining):
    def __init__(self, x, y, n_labels, dict_size, dict_dim=100):
        args = dict(
            x=x,
            y=y,
            n_labels=n_labels,
            dict_size=dict_size,
            dict_dim=dict_dim
        )
        super().__init__(**args)
