# Parallel Feature Extraction with CLIP

## Overview

The following scripts are executable.
See `Makefile` for an example.

+ `scripts/prepare_mnist.py` downloads and stores the MNIST dataset in the NPY format.
+ `scripts/apply_clip_to_images.py` preprocess image into embeddings with CLIP.
+ `scripts/mlp_train.py` trains the neural network.
+ `scripts/mlp_apply.py` evaluates the neural network.

As for the supporting files, we have

+ `scripts/config.py` specifies the model and hyperparameters.
+ `Pipfile` and `Pipfile.lock` specifies the exact dependencies we are using.
+ `data/` stores images, embeddings, predictions, and labels.
+ `checkpoints/` stores model parameters.

## Prerequisite

Run `make .venv` to install the dependencies with [Pipenv](https://pipenv.pypa.io/en/latest/).

## Usage

`make`.

## Parallelism

It turns out that PyTorch calls [ATen](https://pytorch.org/cppdocs/#aten) under the hood, which links against [oneDNN](https://github.com/oneapi-src/oneDNN), which does multi-threaded calculation by default.
In particular, matrix multiplication and convolution are both embarrassingly parallel problems.
As a result, all CPU cores will be used even if you don't call any `multiprocessing` primitive in Python.
This gives us two knots to tune: `MKL_NUM_THREADS` to specify the number of threads used by oneDNN subroutines, and `--processes` to specify the number of processed forked.

Here is the time it takes to do feature extraction (i.e. only `scripts/prepare_mnist.py`) under some configurations.
Note that there could be some fluctuations some I did not average the execution time across several replications.

| `MKL_NUM_THREADS` | `--processes` | Execution Time |
| ----------------- | ------------- | -------------- |
| 1                 | 1             | 154.80 mins    |
| 48                | 1             | 795.93 secs    |
| 48                | 2             | 487.98 secs    |
| 1                 | 48            | 505.04 secs    |
| 2                 | 48            | 460.43 secs    |

## Issues

The training accuracy looks reasonable, but somehow the test acuracy is terrible.
Maybe it's because CLIP are not familiar with tiny black-and-white images of size 28x28, but there could be other issues.

```
# Training
Epoch #1, Epoch average loss = 1.590, Epoch average accuracy = 0.550
Epoch #2, Epoch average loss = 0.312, Epoch average accuracy = 0.961
Epoch #3, Epoch average loss = 0.122, Epoch average accuracy = 0.980
Epoch #4, Epoch average loss = 0.071, Epoch average accuracy = 0.989
Epoch #5, Epoch average loss = 0.053, Epoch average accuracy = 0.992
Epoch #6, Epoch average loss = 0.052, Epoch average accuracy = 0.992
Epoch #7, Epoch average loss = 0.032, Epoch average accuracy = 0.995
Epoch #8, Epoch average loss = 0.021, Epoch average accuracy = 0.997
Epoch #9, Epoch average loss = 0.025, Epoch average accuracy = 0.996
Epoch #10, Epoch average loss = 0.016, Epoch average accuracy = 0.998
Epoch #11, Epoch average loss = 0.020, Epoch average accuracy = 0.996
Epoch #12, Epoch average loss = 0.017, Epoch average accuracy = 0.997
Epoch #13, Epoch average loss = 0.015, Epoch average accuracy = 0.997
Epoch #14, Epoch average loss = 0.017, Epoch average accuracy = 0.996
Epoch #15, Epoch average loss = 0.008, Epoch average accuracy = 0.999
Epoch #16, Epoch average loss = 0.015, Epoch average accuracy = 0.997

# Testing
Accuracy: 0.0976
```

Additionally, JAX cannot defect the TPUs for some reason.

```
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
```
