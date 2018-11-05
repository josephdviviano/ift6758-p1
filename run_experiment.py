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
    knn_results = knn(test_mode=True)
    hierarchical_results = hierarchical(test_mode=True)

    import IPython; IPython.embed()

if __name__ == '__main__':
    main()

