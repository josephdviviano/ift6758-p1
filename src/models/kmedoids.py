import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.models.helper import assignment
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin


def cluster(distance_matrix, n_clusters):
    m = distance_matrix.shape[0]  # number of points

    # Pick k  medoids.
    print('Generating initial medoids...')
    curr_medoids = generate_initial_medoids(distance_matrix, n_clusters)
    print('Finished generating initial medoids!')

    old_medoids = np.array([-1] * n_clusters)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * n_clusters)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distance_matrix)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distance_matrix)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

def generate_initial_medoids(distance_matrix, n_clusters):
    mask = np.ma.masked_array(distance_matrix, np.zeros(distance_matrix.shape))
    arg_medoid = random.randint(0, distance_matrix.shape[1]-1)
    mask.mask[:, arg_medoid] = True
    medoids = [arg_medoid]
    for i in range(n_clusters-1):
        arg_medoid = mask[arg_medoid, :].argmax(fill_value=0)
        medoids.append(arg_medoid)
        mask.mask[:, arg_medoid] = True
    medoids = np.array(medoids)
    return medoids

def assign_points_to_clusters(medoids, distance_matrix):
    distances_to_medoids = distance_matrix[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distance_matrix):
    mask = np.ones(distance_matrix.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distance_matrix, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


class KMedoids(BaseEstimator, ClassifierMixin):
    """KMedoids method for clustering"""

    def __init__(self, n_clusters=10, metric='euclidean'):

        self.n_clusters = n_clusters
        self.metric = metric

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert (type(self.n_clusters) == int), "n_clusters parameter must be integer"

        print('Generating distance matrix...')
        distance_matrix = pairwise_distances(X, metric=self.metric)
        print('Finished generating distance matrix!')

        print('Generating k-medoids instance and processing...')
        self.labels_, self.cluster_centers_ = cluster(distance_matrix, self.n_clusters)
        print('Finished k-medoids!')

        self.cluster_medoids_ = X[self.cluster_centers_]
        self.transformation_ = assignment(self.labels_, y)

        return self

    def _predict_single(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning\
        distances_to_cluster_medoids = [pairwise_distances(i.reshape((-1, 1)).T, x, metric=self.metric) for i in self.cluster_medoids_]
        return self.transformation_[self.cluster_centers_[np.argmin(distances_to_cluster_medoids)]]

    def predict(self, X):
        try:
            getattr(self, "cluster_centers_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return [self._predict_single(x.reshape((1, -1))) for x in X]

    # def score(self, X, y=None):
    #     # counts number of values bigger than mean
    #     return(sum(self.predict(X)))


if __name__ == '__main__':
    np.random.seed(5)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    estimators = [('k_medoids_iris_8', KMedoids(n_clusters=3)),
                  ('k_medoids_iris_3', KMedoids(n_clusters=3, metric='cosine')),]

    fignum = 1
    titles = ['Euclidean Distance', 'Cosine Similarity']
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X, y)
        accuracy = accuracy_score(y, est.predict(X))
        fig.text(-0.8, 0.9, 'Accuracy: {}'.format(accuracy))
        labels = est.labels_
        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1]+', Accuracy: {0:.2f}'.format(accuracy))
        ax.dist = 12
        fignum = fignum + 1
        fig.show()

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean(),
                  X[y == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12

    fig.show()

