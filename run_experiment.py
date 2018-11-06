#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import sklearn

from src.models.knn import knn
from src.models.hierarchical import hierarchical

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():

    test = True
    knn_results_img = knn(test_mode=test)
    knn_results_vec = knn(test_mode=test, custom_data=True)
    hierarchical_results_img = hierarchical(test_mode=test)
    hierarchical_results_vec = hierarchical(test_mode=test, custom_data=True)

    print('***done!***')
    import IPython; IPython.embed()

if __name__ == '__main__':
    main()

