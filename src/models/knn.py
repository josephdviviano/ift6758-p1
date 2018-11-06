
from copy import copy
from pathlib import Path
from scipy.stats import randint
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.pipeline import Pipeline
from src.models.helper import convert_ds_to_np, cosine
import numpy as np
import os
import torch



def knn(test_mode=False, custom_data=False):

    results = {
        'test':  {'accuracy': None, 'confusion': None},
        'train': {'accuracy': [], 'confusion': []},
        'best_model': None, 'best_acc' : 0
    }

    settings = {
        'cv_iter': 100,
        'cv_score': 'accuracy',
        'n_cv': 3,
        'n_folds': 10,
        'n_samples': 2000,
    }

    if test_mode:
        settings['n_samples'] = 100

    data_path = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed')

    if custom_data:
        data = np.load(os.path.join(data_path, 'vectors.npy'))
        X = data.item()['data']
        y = data.item()['labels']

        X_train = X[:60000, :]
        X_test = X[60000:, :]
        y_train = y[:60000]
        y_test = y[60000:]
        del X, y, data

        metric = cosine

    else:
        train_data = os.path.join(data_path, 'training.pt')
        test_data = os.path.join(data_path, 'test.pt')
        X_train, y_train = convert_ds_to_np(train_data)
        X_test, y_test = convert_ds_to_np(train_data)

        metric = 'euclidean'

    X_train = X_train[:settings['n_samples'], :]
    y_train = y_train[:settings['n_samples']]
    X_test = X_test[:settings['n_samples'], :]
    y_test = y_test[:settings['n_samples']]

    # model set up using pipeline for randomized CV
    clf = knc(metric=metric, algorithm='brute')

    cv_opts = {
        'n_neighbors': randint(2,10)
    }

    model = RandomizedSearchCV(
        clf, cv_opts, n_jobs=-1, n_iter=settings['cv_iter'],
        cv=settings['n_cv'], scoring=settings['cv_score']
    )

    kf = StratifiedKFold(n_splits=settings['n_folds'], shuffle=True)

    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        X_trn = X_train[train_idx]
        X_vld = X_train[valid_idx]
        y_trn = y_train[train_idx]
        y_vld = y_train[valid_idx]

        model.fit(X_trn, y_trn)

        y_pred = model.predict(X_vld)
        this_acc = accuracy_score(y_pred, y_vld)
        results['train']['accuracy'].append(this_acc)
        results['train']['confusion'].append(confusion_matrix(y_pred, y_vld))

        print('[{}/{}]: this={} : best={}'.format(
            i+1, settings['n_folds'], this_acc, results['best_acc']))
        if this_acc > results['best_acc']:
            results['best_acc'] = this_acc
            results['best_model'] = copy(model)

    # get test performance with best model:
    y_pred = results['best_model'].predict(X_test)
    results['test']['accuracy'] = accuracy_score(y_pred, y_test)
    results['test']['confusion'] = confusion_matrix(y_pred, y_test)

    return(results)


