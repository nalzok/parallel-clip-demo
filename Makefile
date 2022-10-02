.PHONY: all clean

all: data/t10k_predictions.npy

data/t10k_predictions.npy: checkpoints/checkpoint_16 data/t10k_embeddings.npy data/t10k_labels.npy
	pipenv run python3 scripts/mlp_apply.py \
		--ckpt_in checkpoints \
		--embedding_in data/t10k_embeddings.npy \
		--ground_truth_in data/t10k_labels.npy \
		--label_out data/t10k_predictions.npy

checkpoints/checkpoint_16: data/train_embeddings.npy data/train_labels.npy
	mkdir -p checkpoints
	pipenv run python3 scripts/mlp_train.py \
		--embedding_in data/train_embeddings.npy \
		--label_in data/train_labels.npy \
		--ckpt_out checkpoints

data/train_embeddings.npy: data/train_images.npy
	pipenv run python3 scripts/apply_clip_to_images.py \
		--image_in data/train_images.npy \
		--embedding_out data/train_embeddings.npy

data/t10k_embeddings.npy: data/t10k_images.npy
	pipenv run python3 scripts/apply_clip_to_images.py \
		--image_in data/t10k_images.npy \
		--embedding_out data/t10k_embeddings.npy

data/train_images.npy data/train_labels.npy data/t10k_images.npy data/t10k_labels.npy:
	mkdir -p data
	pipenv run python3 scripts/prepare_mnist.py

.venv: Pipfile Pipfile.lock
	PIPENV_VENV_IN_PROJECT=1 pipenv install

clean:
	rm -f data/t10k_predictions.npy checkpoints/checkpoint_*

distclean:
	rm -rf data checkpoints
