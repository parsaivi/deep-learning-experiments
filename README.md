# Deep learning experiments

Three small experiments from the Deep Learning course at Sharif University of Technology (Fall 2026). Each notebook keeps its outputs, so you can read the results without running anything.

## 1. A two-layer network and backpropagation in NumPy

[`01-backprop-numpy/backprop_two_moons.ipynb`](01-backprop-numpy/backprop_two_moons.ipynb)

A `2 → 8 → 1` network with its backward pass derived and written by hand, trained on two moons and checked against finite differences and PyTorch autograd. Then one change at a time: hidden width, the hidden nonlinearity, the learning rate, and a second hidden layer.

- Every run first stalls near the loss of the best straight line (about 0.29) before it bends its boundary. Without the hidden tanh it never leaves that plateau, because the network collapses to logistic regression.
- A second hidden layer reaches 99% test accuracy against 96.5% for one layer at the same learning rate, but one layer reaches 100% with a larger learning rate. The gap is about how fast each model is optimized.

## 2. Optimizers on small 2D problems

[`02-optimizers-2d/optimizers_2d.ipynb`](02-optimizers-2d/optimizers_2d.ipynb)

GD, Newton, momentum, Nesterov, Adagrad, RMSProp, Adam and mini-batch SGD with learning-rate schedules, each run on a 2D problem picked to expose one difficulty: ill-conditioning, a saddle, the Rosenbrock valley, and noisy gradients.

- Newton converges to the saddle from every start with |y| < 0.45, where GD rolls down to a minimum.
- Rotating an ill-conditioned quadratic by 45° doubles the steps Adam needs (57 → 122); GD and Newton do not notice the rotation.
- SGD on data sorted by target ends 1.6 away from the minimum, against 0.12 when the data is reshuffled every epoch.

## 3. Regularization on a small-data MNIST-1D problem

[`03-regularization-mnist1d/regularization_mnist1d.ipynb`](03-regularization-mnist1d/regularization_mnist1d.ipynb)

An MLP trained on 1000 MNIST-1D examples reaches 100% train and 45.8% validation accuracy. The notebook measures bias and variance directly, then tries L2, early stopping, dropout written from scratch, ensembles, bagging, augmentation and label smoothing one at a time, and ends with implicit regularization of SGD and double descent.

- Augmentation helps most (validation accuracy 0.458 → 0.642), then dropout (0.537) and L2 (0.514).
- Label smoothing barely changes accuracy but cuts the calibration error from 0.42 to 0.036.
- With 20% label noise, validation error peaks where the network can just fit the training set and falls again for wider networks.
- With 500 training examples, a small 1D CNN with augmentation, dropout and a 5-seed ensemble reaches 83.8% test accuracy.

## Running

All three run on a CPU. The third installs `mnist1d` on first run and takes about 10 minutes.

```bash
pip install -r requirements.txt
jupyter notebook
```

## Credit

The course staff designed the original exercises, the test problems, and the data-loading, helper, plotting and checking code: Ali Torkashvand and Farzan Rahmani (notebook 1), Raya Namazi (notebook 2) and Mohammadreza Monemian (notebook 3). I implemented the models, optimizers and regularizers, ran the experiments and wrote the analysis.
