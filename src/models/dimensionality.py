import numpy as np
import torch
import os
from pathlib import Path
from sklearn import manifold
import matplotlib.pyplot as plt
from knn import knn
from scipy.spatial import distance


def convert_ds_to_np(D):
    """torch ds --> numpy matrix. x becomes a (n x m) matrix."""
    X, y = torch.load(D)
    X = X.numpy().swapaxes(1,3).squeeze()            # make (n x row x col)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]) # make (n x m)

    return X, y.numpy()


TRAIN_DATA = os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
TEST_DATA = os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')
X_train, y_train = convert_ds_to_np(TRAIN_DATA)
X_test, y_test = convert_ds_to_np(TEST_DATA)

data_features = np.load('../../data/processed/vectors.npy').item()
features = data_features['data'][60000:62000, :]
labels = data_features['labels'][60000:62000]

mnist = X_test[0:2000, :]
mnist_labels = y_test[0:2000]

MDS = manifold.MDS(n_jobs=4)
mds_features = MDS.fit_transform(features)
mds_mnist = MDS.fit_transform(mnist)
Isomap = manifold.Isomap(n_jobs=4)

isomap_features = Isomap.fit_transform(features)
isomap_mnist = Isomap.fit_transform(mnist)

fig, ax = plt.subplots()
for g in np.unique(mnist_labels):
    ix = np.where(mnist_labels == g)
    ax.scatter(mds_features[ix, 0], mds_features[ix, 1], label=g)
ax.legend()
plt.title('PCoA using LeNet features')
plt.show()
fig, ax = plt.subplots()


for g in np.unique(mnist_labels):
    ix = np.where(mnist_labels == g)
    ax.scatter(mds_mnist[ix, 0], mds_mnist[ix, 1], label=g)
plt.title('PCoA using raw images')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for g in np.unique(mnist_labels):
    ix = np.where(mnist_labels == g)
    ax.scatter(isomap_mnist[ix, 0], isomap_mnist[ix, 1], label=g)
plt.title('Isomap using raw images')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for g in np.unique(mnist_labels):
    ix = np.where(mnist_labels == g)
    ax.scatter(isomap_features[ix, 0], isomap_features[ix, 1], label=g)
plt.title('Isomap using LeNet features')
ax.legend()
plt.show()

