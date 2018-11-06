from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch


def cosine(x, y):
    """cosine distance between two 1-d vectors"""
    return(1 - (x.dot(y.T) / (norm(x) + norm(y))))


def assignment(assigned_clusters, correct_labels):
    """duplicate of match_labels"""
    assert assigned_clusters.shape == correct_labels.shape
    unique_classes = np.unique(correct_labels)
    unique_clusters = np.unique(assigned_clusters)
    cost_matrix = np.zeros((len(unique_classes), len(unique_clusters)))
    for i in range(len(unique_classes)):
        boolean_array_labels = correct_labels == unique_classes[i]
        clusters_in_label = assigned_clusters[boolean_array_labels]
        for j in range(len(unique_clusters)):
            cost_matrix[i, j] = len(clusters_in_label[clusters_in_label == unique_clusters[j]])
    print(cost_matrix)
    cost_matrix = 1 - cost_matrix/np.sum(cost_matrix, axis=0)
    classes_index, clusters_index = linear_sum_assignment(cost_matrix)
    transformation = dict(zip(unique_clusters[list(clusters_index)], unique_classes[list(classes_index)]))
    return transformation


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

    # assign values in b so they resemble a (as much as is possible)
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


if __name__ == '__main__':
    a = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    b = np.array([37, 37, 37, 99, 150, 150, 150, 99, 99, 120, 120, 120, 150, 99, 99, 150])
    print(assignment(b, a))
