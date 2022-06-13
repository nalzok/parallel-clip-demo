import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
import click

from config import config


@click.command()
@click.option('--ckpt_in', type=click.Path(exists=True))
@click.option('--embedding_in', type=click.File('rb'))
@click.option('--ground_truth_in', type=click.File('rb'))
@click.option('--label_out', type=click.File('wb'))
def evaluator(ckpt_in, embedding_in, ground_truth_in, label_out):
    model = config.model
    batch_size = config.batch_size

    embeddings = jnp.array(np.load(embedding_in))

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.empty(embeddings.shape[1:]))

    tx = optax.adam(learning_rate=config.lr)    # not really necessary
    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=tx)
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_in, target=state)

    @jax.jit
    def eval_step(state: train_state.TrainState, embedding_batch, predictions):
        logits = model.apply(state.params, embedding_batch)
        predictions_batch = jnp.argmax(logits, axis=1)
        return predictions.at[i:i+batch_size].set(predictions_batch)

    predictions = jnp.empty((embeddings.shape[0],))
    steps = (embeddings.shape[0] + batch_size - 1) // batch_size
    for i in range(steps):
        embedding_batch = embeddings[i:i+batch_size]
        predictions = eval_step(state, embedding_batch, predictions)

    np.save(label_out, np.asarray(predictions))

    if ground_truth_in is not None:
        ground_truth = np.load(ground_truth_in)
        print('Accuracy:', np.mean(predictions == ground_truth))


if __name__ == '__main__':
    evaluator()
