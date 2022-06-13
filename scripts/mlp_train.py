import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
import click
from functools import partial

from config import config

@click.command()
@click.option('--embedding_in', type=click.File('rb'))
@click.option('--label_in', type=click.File('rb'))
@click.option('--ckpt_out', type=click.Path(exists=True))
def trainer(embedding_in, label_in, ckpt_out):
    model = config.model
    epochs = config.epochs
    batch_size = config.batch_size
    lr = config.lr

    embeddings = jnp.array(np.load(embedding_in))
    labels = jnp.array(np.load(label_in))
    assert embeddings.shape[0] == labels.shape[0]

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.empty(embeddings.shape[1:]))

    tx = optax.adam(learning_rate=lr)
    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=tx)

    @partial(jax.value_and_grad, has_aux=True)
    def metrics(params, embedding_batch, label_batch):
        logits = model.apply(params, embedding_batch)
        loss = jnp.mean(jax.vmap(softmax_cross_entropy_with_integer_labels)(logits, label_batch))
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == label_batch)
        return loss, accuracy

    @jax.jit
    def train_step(state: train_state.TrainState, embedding_batch, label_batch):
        (loss_batch, accuracy_batch), grads = metrics(state.params, embedding_batch, label_batch)
        state = state.apply_gradients(grads=grads)
        return (loss_batch, accuracy_batch), state

    steps = (embeddings.shape[0] + batch_size - 1) // batch_size
    for epoch in range(1, epochs + 1):
        key, data_key = jax.random.split(key)
        embeddings = jax.random.permutation(data_key, embeddings)
        labels = jax.random.permutation(data_key, labels)

        loss_epoch = 0
        accuracy_epoch = 0
        for i in range(steps):
            embedding_batch, label_batch = embeddings[i:i+batch_size], labels[i:i+batch_size]
            (loss_batch, accuracy_batch), state = train_step(state, embedding_batch, label_batch)
            loss_epoch += loss_batch    # FIXME: weight by batch size in case embeddings.shape[0] % batch_size != 0
            accuracy_epoch += accuracy_batch
        print(f'Epoch #{epoch}, Epoch average loss = {loss_epoch/steps:.3f}, Epoch average accuracy = {accuracy_epoch/steps:.3f}')

        checkpoints.save_checkpoint(ckpt_dir=ckpt_out, target=state, step=epoch)


# TODO: replace this with optax.softmax_cross_entropy_with_integer_labels when they cut a new release
def softmax_cross_entropy_with_integer_labels(logits, labels):
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return log_normalizers - label_logits


if __name__ == '__main__':
    trainer()
