import os
from pathlib import Path
from sklearn.metrics import accuracy_score
from src.models.kmedoids import KMedoids
from src.models.helper import convert_ds_to_np


if __name__ == '__main__':

    TRAIN_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(
        Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    X_train, y_train = convert_ds_to_np(TRAIN_DATA)
    X_test, y_test = convert_ds_to_np(TEST_DATA)

    # K-Medoids
    kmedoids_euclidean = KMedoids()
    kmedoids_cosine = KMedoids(metric='cosine')

    kmedoids_cosine.fit(X_train, y_train)
    predictions_kmedoids_cosine = kmedoids_cosine.predict(X_test)
    accuracy_kmedoids_cosine = accuracy_score(y_test, predictions_kmedoids_cosine)
    print('The accuracy of the cosine similarity in the feature space is {}'.format(accuracy_kmedoids_cosine))