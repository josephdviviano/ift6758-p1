
from copy import copy
from pathlib import Path
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.pipeline import Pipeline
import numpy as np
import os
import torch


SETTINGS = {
    'cv_iter': 100,
    'cv_score': 'f1_macro',
    'n_cv': 3,
    'n_folds': 10
}


def convert_ds_to_np(D):
    """torch ds --> numpy matrix. x becomes a (n x m) matrix."""
    X, y = torch.load(D)
    X = X.numpy().swapaxes(1,3).squeeze()            # make (n x row x col)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]) # make (n x m)

    return(X, y.numpy())


def knn():

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    X_test, y_test = convert_ds_to_np(TEST_DATA)

    # model set up using pipeline for randomized CV
    clf = knc(metric='euclidean')

    settings = {
        'n_neighbors': randint(2,10)
    }

    model = RandomizedSearchCV(
        clf, settings, n_jobs=-1, n_iter=SETTINGS['cv_iter'],
        cv=SETTINGS['n_cv'], scoring=SETTINGS['cv_score']
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

