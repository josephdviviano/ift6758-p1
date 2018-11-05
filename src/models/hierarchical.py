
from copy import copy
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
import os
import torch

SETTINGS = {
    'n_repetitions': 10
}

RESULTS = {
    'test':  {'accuracy': [], 'confusion': []},
    'train': {'accuracy': [], 'confusion': []},
    'best_model': None, 'best_acc' : 0
}


def match_labels(a, b):
    """
    Reorders a to match b, assuming they are two sets of labels, with the same
    number of labels, but they are not nessicarily the same.

    Minimizes the distance between the input label vectors. Inputs must be the
    same length. For details see section 2 of Lange T et al. 2004.

    E.g,
    a = [1,1,2,3,3,4,4,4,2]
    b = [2,2,3,1,1,4,4,4,3]
    optimal: 1 -> 2; 2 -> 3; 3 -> 1; 4 -> 4

    Inspired by http://things-about-r.tumblr.com/post/36087795708/matching-clustering-solutions-using-the-hungarian
    """
    a = np.array(a)
    b = np.array(b)
    ids_a = np.unique(a)
    ids_b = np.unique(b)

    assert len(a) == len(b)
    assert len(ids_a) == len(ids_b)

    # construct a distance matrix D between a and b
    n = len(ids_a)
    D = np.zeros((n, n)) # distance matrix

    for x in np.arange(n):
        for y in np.arange(n):
            idx_a = np.where(a == x)[0]
            idx_b = np.where(b == y)[0]
            n_int = len(np.intersect1d(idx_a, idx_b))

            # distance = (# in label) - 2*sum(# in intersection)
            D[x,y] = (len(idx_a) + len(idx_b) - 2*n_int)

    # permute labels w/ minimum weighted bipartite matching (hungarian method)
    idx_D_x, idx_D_y = linear_sum_assignment(D)
    b_out = np.zeros(len(b))

    for i, c in enumerate(idx_D_y):
        idx_map = np.where(b == c)
        b_out[idx_map] = i

    return(b_out)


def convert_ds_to_np(D):
    """torch ds --> numpy matrix. x becomes a (n x m) matrix."""
    X, y = torch.load(D)
    X = X.numpy().swapaxes(1,3).squeeze()            # make (n x row x col)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]) # make (n x m)

    return(X, y.numpy())


def hierarchical(test_mode=False):

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    # merge X and y for unsupervised case
    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    X_test, y_test = convert_ds_to_np(TEST_DATA)
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if test_mode:
        X = X[:100, :]
        y = y[:100]

    model = hc(n_clusters=10, affinity ='euclidean') # 'precomputed'


    for i in range(SETTINGS['n_repetitions']):
        model.fit(X)

        y_pred = model.labels_
        y_pred = match_labels(y, y_pred)

        this_acc = accuracy_score(y_pred, y)
        RESULTS['test']['accuracy'].append(this_acc)
        RESULTS['test']['confusion'].append(confusion_matrix(y_pred, y))

        if this_acc > RESULTS['best_acc']:
            print('[{}/{}]: new model found {}/{}'.format(
                i+1, SETTINGS['n_repetitions'], this_acc, RESULTS['best_acc']))
            RESULTS['best_acc'] = this_acc
            RESULTS['best_model'] = copy(model)

    return(RESULTS)


