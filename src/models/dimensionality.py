import numpy as np
import torch
import os
from pathlib import Path
from sklearn import manifold
import matplotlib.pyplot as plt
# from knn import knn
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
features_test = data_features['data'][60000:62000, :]
features_train = data_features['data'][0:2000, :]

mnist_train = X_train[0:2000, :]
train_labels = y_train[0:2000]
mnist_test = X_test[0:2000, :]
test_labels = y_test[0:2000]


MDS = manifold.MDS(n_jobs=4)
mds_features = MDS.fit_transform(features_test)
mds_mnist = MDS.fit_transform(mnist_test)

knn = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30])
isomaps_features = []
isomaps_mnist = []
# isomap
for i in range(0, len(knn)):
    Isomap = manifold.Isomap(n_neighbors=knn[i], n_jobs=4)

    isomap_features = Isomap.fit_transform(features_train)
    isomap_features_test = Isomap.transform(features_test)
    isomaps_features.append(isomap_features_test)

    isomap_mnist = Isomap.fit_transform(mnist_train)
    isomap_mnist_test = Isomap.transform(mnist_test)
    isomaps_mnist.append(isomap_mnist_test)


# plots
fig, ax = plt.subplots()
for g in np.unique(test_labels):
    ix = np.where(test_labels == g)
    ax.scatter(mds_features[ix, 0], mds_features[ix, 1], label=g)
ax.legend()
plt.title('PCoA using LeNet features')
# plt.savefig('pcoa_features.png')
plt.show()
fig, ax = plt.subplots()
#
for g in np.unique(test_labels):
    ix = np.where(test_labels == g)
    ax.scatter(mds_mnist[ix, 0], mds_mnist[ix, 1], label=g)
plt.title('PCoA using raw images')
ax.legend()
# plt.savefig('pcoa_mnist.png')
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
j = 0
for i, ax in enumerate(fig.axes):
    this_fts = isomaps_features[j]
    for g in np.unique(test_labels):
        ix = np.where(test_labels == g)
        ax.scatter(this_fts[ix, 0], this_fts[ix, 1], label=g)
    ax.set_title('k = ' + str(knn[j]))
    j += 1
    ax.legend()
plt.tight_layout()
# plt.savefig('features.png')
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
j = 0
for i, ax in enumerate(fig.axes):
    this_fts = isomaps_mnist[j]
    for g in np.unique(test_labels):
        ix = np.where(test_labels == g)
        ax.scatter(this_fts[ix, 0], this_fts[ix, 1], label=g)
    ax.set_title('k = ' + str(knn[j]))
    j += 1
    ax.legend()
plt.tight_layout()
# plt.savefig('mnist.png')
plt.show()





