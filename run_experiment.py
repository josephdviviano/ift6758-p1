#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import sklearn

from src.models.knn import knn
from src.models.hierarchical import hierarchical

def main():
    #best_model, test_score = knn(test_mode=True)
    hierarchical(test_mode=True)

if __name__ == '__main__':
    main()

