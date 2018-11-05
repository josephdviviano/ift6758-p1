import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment(assigned_clusters, correct_labels):
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


if __name__ == '__main__':
    a = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    b = np.array([37, 37, 37, 99, 150, 150, 150, 99, 99, 120, 120, 120, 150, 99, 99, 150])
    print(assignment(b, a))