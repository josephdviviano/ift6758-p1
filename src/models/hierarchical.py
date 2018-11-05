
from copy import copy
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
import os
import torch

SETTINGS = {
    'cv_iter': 100,
    'cv_score': 'f1_macro',
    'n_cv': 3,
    'n_folds': 10
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
            idx_a = np.where(a-1 == x)[0]
            idx_b = np.where(b-1 == y)[0]
            n_int = len(np.intersect1d(idx_a, idx_b))
            # distance = (# in cluster) - 2*sum(# in intersection)
            D[x,y] = (len(idx_a) + len(idx_b) - 2*n_int)

    # permute labels w/ minimum weighted bipartite matching (hungarian method)
    idx_D_x, idx_D_y = linear_sum_assignment(D)
    b_out = np.zeros(len(b))

    for i, c in enumerate(idx_D_y):
        idx_map = np.where(b == c+1) # +1 to keep indicies aligned
        b_out[idx_map] = i+1

    return(b_out)


def convert_ds_to_np(D):
    """torch ds --> numpy matrix. x becomes a (n x m) matrix."""
    X, y = torch.load(D)
    X = X.numpy().swapaxes(1,3).squeeze()            # make (n x row x col)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]) # make (n x m)

    return(X, y.numpy())


def hierarchical():

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    X_test, y_test = convert_ds_to_np(TEST_DATA)

    clf = hc(n_clusters=10, affinity ='euclidean') # 'precomputed'

    settings = {
        'linkage': ['ward', 'complete', 'average', 'single']
    }

    model = GridSearchCV(
        clf, settings, n_jobs=-1, scoring=SETTINGS['cv_score']
    )

    kf = StratifiedKFold(n_splits=SETTINGS['n_folds'], shuffle=True)

    results = {
        'test':  {'loss': [], 'accuracy': [], 'confusion': [], 'errors': []},
        'train': {'loss': [], 'accuracy': [], 'confusion': []},
        'cv': {}
    }

    best_model = None
    best_score = 0
    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        X_trn = X_train[train_idx]
        X_vld = X_train[valid_idx]
        y_trn = y_train[train_idx]
        y_vld = y_train[valid_idx]

        model.fit(X_trn, y_trn)

        import IPython; IPython.embed()

        y_pred = model.predict(X_vld)
        this_score = accuracy_score(y_pred, y_vld)

        if this_score > best_score:
            print('[{}/{}]: new model found {}/{}'.format(
                i+1, SETTINGS['n_folds'], this_score, best_score))
            best_score = this_score
            best_model = copy(model)


    # get test performance with best model:
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_pred, y_test)

    return(best_model, test_score)

