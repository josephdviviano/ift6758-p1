
from copy import copy
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.models.helper import convert_ds_to_np, match_labels
import numpy as np
import os
import torch


def hierarchical(test_mode=False, custom_data=False):

    data_path = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed')

    results = {
        'test':  {'accuracy': [], 'confusion': []},
        'best_model': None, 'best_acc' : 0, 'lost_subjects' : 0
    }

    settings = {'n_repetitions': 10, 'n_samples': 2000}

    if test_mode:
        settings['n_samples'] = 100

    if custom_data:
        data = np.load(os.path.join(data_path, 'vectors.npy'))
        X = data.item()['data']
        y = data.item()['labels']

        del data

        affinity = 'cosine'
        linkage = 'average'

    else:
        train_data = os.path.join(data_path, 'training.pt')
        test_data = os.path.join(data_path, 'test.pt')

        # merge X and y for unsupervised case
        X_train, y_train = convert_ds_to_np(train_data)
        X_test, y_test = convert_ds_to_np(test_data)
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        affinity = 'euclidean'
        linkage = 'average'

    model = hc(n_clusters=10, affinity=affinity, linkage=linkage)

    # remove subject with no activation values
    bad_idx = np.where(np.sum(X, axis=1) == 0)[0]
    idx = np.setdiff1d(np.arange(X.shape[0]), bad_idx)
    n = len(idx) # number of remaining subjects

    X = X[idx, :]
    y = y[idx]

    results['lost_subjects'] = len(bad_idx)

    for i in range(settings['n_repetitions']):

        # random sampling of dataset
        idx = np.arange(n)
        np.random.shuffle(idx)

        X_samp = X[idx[:settings['n_samples']], :]
        y_samp = y[idx[:settings['n_samples']]]

        model.fit(X_samp)

        y_pred = model.labels_
        y_pred = match_labels(y_samp, y_pred)

        this_acc = accuracy_score(y_pred, y_samp)
        results['test']['accuracy'].append(this_acc)
        results['test']['confusion'].append(confusion_matrix(y_pred, y_samp))

        print('[{}/{}]: this={} : best={}'.format(
            i+1, settings['n_repetitions'], this_acc, results['best_acc']))
        if this_acc > results['best_acc']:
            results['best_acc'] = this_acc
            results['best_model'] = copy(model)

    return(results)


