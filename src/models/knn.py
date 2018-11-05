
from copy import copy
from pathlib import Path
from scipy.stats import randint
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.pipeline import Pipeline
import numpy as np
import os
import torch


SETTINGS = {
    'cv_iter': 100,
    'cv_score': 'accuracy',
    'n_cv': 3,
    'n_folds': 10
}

RESULTS = {
    'test':  {'accuracy': None, 'confusion': None},
    'train': {'accuracy': [], 'confusion': []},
    'best_model': None, 'best_acc' : 0
}


def convert_ds_to_np(D):
    """torch ds --> numpy matrix. x becomes a (n x m) matrix."""
    X, y = torch.load(D)
    X = X.numpy().swapaxes(1,3).squeeze()            # make (n x row x col)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]) # make (n x m)

    return(X, y.numpy())


def knn(test_mode=False):

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    X_test, y_test = convert_ds_to_np(TEST_DATA)

    if test_mode:
        X_train = X_train[:100, :]
        y_train = y_train[:100]
        X_test = X_test[:100, :]
        y_test = y_test[:100]

    # model set up using pipeline for randomized CV
    clf = knc(metric='euclidean')

    cv_opts = {
        'n_neighbors': randint(2,10)
    }

    model = RandomizedSearchCV(
        clf, cv_opts, n_jobs=-1, n_iter=SETTINGS['cv_iter'],
        cv=SETTINGS['n_cv'], scoring=SETTINGS['cv_score']
    )

    kf = StratifiedKFold(n_splits=SETTINGS['n_folds'], shuffle=True)

    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        X_trn = X_train[train_idx]
        X_vld = X_train[valid_idx]
        y_trn = y_train[train_idx]
        y_vld = y_train[valid_idx]

        model.fit(X_trn, y_trn)

        y_pred = model.predict(X_vld)
        this_acc = accuracy_score(y_pred, y_vld)
        RESULTS['train']['accuracy'].append(this_acc)
        RESULTS['train']['confusion'].append(confusion_matrix(y_pred, y_vld))

        if this_acc > RESULTS['best_acc']:
            print('[{}/{}]: new model found {}/{}'.format(
                i+1, SETTINGS['n_folds'], this_acc, RESULTS['best_acc']))
            RESULTS['best_acc'] = this_acc
            RESULTS['best_model'] = copy(model)


    # get test performance with best model:
    y_pred = RESULTS['best_model'].predict(X_test)
    RESULTS['test']['accuracy'] = accuracy_score(y_pred, y_test)
    RESULTS['test']['confusion'] = confusion_matrix(y_pred, y_test)

    return(RESULTS)


