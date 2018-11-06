#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import sklearn
import pandas as pd
import seaborn as sns

from src.models.knn import knn
from src.models.hierarchical import hierarchical

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_boxplot(a, b, title, output):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame()
    df['experiment'] = ['MNIST Image']*10 + ['MNIST CNN Embeddings']*10
    df['accuracy'] = a['accuracy'] + b['accuracy']

    plot = sns.boxplot(x='experiment', y='accuracy', data=df)
    plot.set_xlabel('Experiment')
    plot.set_ylabel('Accuracy')
    plot.set_title(title)
    fig = plot.get_figure()
    fig.savefig(output)
    fig.clear()


def make_confusion_mat(mat, title, output):
    import seaborn as sns
    plot = sns.heatmap(mat, annot=True, fmt="d", linewidths=0.5, cmap='Blues')
    plot.set_xticklabels(np.arange(10)+1)
    plot.set_yticklabels(np.arange(10)+1)
    plot.set_xlabel('MNIST Digit')
    plot.set_ylabel('MNIST Digit Predicted')
    plot.set_title(title)
    fig = plot.get_figure()
    fig.savefig(output)
    fig.clear()

def plot_dendrogram(model, **kwargs):
    """plots a dendrogram using scipy, NOT USED, NOT USEFUL"""
    from scipy.cluster.hierarchy import dendrogram

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def main():

    test = False
    knn_results_img = knn(test_mode=test)
    knn_results_vec = knn(test_mode=test, custom_data=True)
    hierarchical_results_img = hierarchical(test_mode=test)
    hierarchical_results_vec = hierarchical(test_mode=test, custom_data=True)

    print('***done!***')

    imgdir = 'report/figures'
    filedir = 'report'

    # knn results
    make_boxplot(knn_results_img['train'], knn_results_vec['train'],
        '10-fold Cross Validation Performance',
        os.path.join(imgdir, 'knn_validation_acc.jpg'))

    make_confusion_mat(knn_results_img['test']['confusion'],
        'MNIST Image Test Confusion Matrix',
        os.path.join(imgdir, 'knn_test_img_confusion.jpg'))

    make_confusion_mat(knn_results_vec['test']['confusion'],
        'MNIST CNN Embeddings Test Confusion Matrix',
        os.path.join(imgdir, 'knn_test_vec_confusion.jpg'))

    with open(os.path.join(filedir, 'knn_stats.csv'), 'w') as f:
        f.write('img,vec\n{},{}'.format(
            knn_results_img['test']['accuracy'],
            knn_results_vec['test']['accuracy'])
        )

    # hierarchical results
    make_boxplot(hierarchical_results_img['test'], hierarchical_results_vec['test'],
        '10-randomized Permutations Classification Performance',
        os.path.join(imgdir, 'hier_test_acc.jpg'))

    idx = np.where(hierarchical_results_img['test']['accuracy'] == np.max(hierarchical_results_img['test']['accuracy']))[0][0]
    make_confusion_mat(hierarchical_results_img['test']['confusion'][idx],
        'MNIST Image Test Confusion Matrix (Best Case)',
        os.path.join(imgdir, 'hier_test_img_confusion.jpg'))

    idx = np.where(hierarchical_results_vec['test']['accuracy'] == np.max(hierarchical_results_vec['test']['accuracy']))[0][0]
    make_confusion_mat(hierarchical_results_vec['test']['confusion'][idx],
        'MNIST CNN Embeddings Test Confusion Matrix (Best Case)',
        os.path.join(imgdir, 'hier_test_vec_confusion.jpg'))

    with open(os.path.join(filedir, 'hier_stats.csv'), 'w') as f:
        f.write('lost_subject\n{}'.format(
            hierarchical_results_vec['lost_subjects'])
        )

    import IPython; IPython.embed()

if __name__ == '__main__':
    main()

