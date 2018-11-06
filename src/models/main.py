import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from src.models.kmedoids import KMedoids
from src.models.helper import convert_ds_to_np


if __name__ == '__main__':

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    SUBSAMPLE = 0.40
    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    train_subsample = np.random.choice(len(X_train), int(len(X_train)*SUBSAMPLE), replace=False)
    X_train = X_train[train_subsample]
    y_train = y_train[train_subsample]
    X_test, y_test = convert_ds_to_np(TEST_DATA)

    data_features = np.load('../../data/processed/vectors.npy').item()
    X_train_features = data_features['data'][:60000][train_subsample]
    y_train_features = data_features['labels'][:60000][train_subsample]
    X_test_features = data_features['data'][60000:]
    y_test_features = data_features['labels'][60000:]

    # K-Medoids
    kmedoids_cosine = KMedoids(metric='cosine')
    kmedoids_cosine.fit(X_train_features, y_train_features)
    predictions_kmedoids_cosine = kmedoids_cosine.predict(X_test_features)
    accuracy_kmedoids_cosine = accuracy_score(y_test_features, predictions_kmedoids_cosine)
    print('The accuracy of the cosine similarity in the feature space is {}'.format(accuracy_kmedoids_cosine))

    kmedoids_euclidean = KMedoids()
    kmedoids_euclidean.fit(X_train, y_train)
    predictions_kmedoids_euclidean = kmedoids_euclidean.predict(X_test)
    accuracy_kmedoids_euclidean = accuracy_score(y_test, predictions_kmedoids_euclidean)
    print('The accuracy of the euclidean distance in the raw image space is {}'.format(accuracy_kmedoids_euclidean))

    # Plots
    ### COSINE ###
    X = X_test_features
    y = y_test_features
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=28, azim=134)
    X = PCA(n_components=3).fit_transform(X)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.set_title("Ground truth in extracted feature space")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

    y = predictions_kmedoids_cosine
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=28, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.set_title("Predictions of K-Medoids with cosine similarity in extracted feature space")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

    ### EUCLIDEAN ###
    X = X_test
    y = y_test
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=28, azim=134)
    X = PCA(n_components=3).fit_transform(X)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.set_title("Ground truth in pixel feature space")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

    y = predictions_kmedoids_euclidean
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=28, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.set_title("Predictions of K-Medoids with euclidean distance in pixel feature space")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()