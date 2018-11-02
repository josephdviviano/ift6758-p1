#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import sklearn

from src.models.knn import knn

def main():
    best_model, test_score = knn()

if __name__ == '__main__':
    main()

