#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import sys
import os
import math
import time
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


import numpy as np
import pandas as pd

# Calculate features

# calculate distance
def calc_dist():
    ...

# calculate velocity
def calc_velocity():
    ...

# calculate velocity
def calc_acceleration():
    ...

# calculate logical array 0000 or 11111, if animal moved in last 10 sec move to 1 else 0
def calc_relative_motion():
    ...


def do_lda(data):
    ...

def do_pca(data):
    mu = data.mean(axis=0)
    data = data - mu
    # data = (data - mu)/data.std(axis=0)  # Uncommenting this reproduces mlab.PCA results
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    print(eigenvectors)


if __name__ == '__main__':

    # Constants
        nfeatures = 5
        feature_coords = 3

        # open file browser
        Tk().withdraw() 
        dataname = askopenfilename() # get filename
        #dataname = 'e3v8103_day_subclipDeepCut_resnet50_ratmovementNov5shuffle1_22500.h5'
        df = pd.read_hdf(dataname)

        # structure of dataframe 
        # scorer    Bodyparts   x   y   likelihood
        #            head
        #            neck
        #            body
        #            rear
        #            tail


        # List columns
        #list(df)

        #df.columns

        print(df.columns.contains)
        # <bound method MultiIndex.__contains__ of MultiIndex(levels=[['DeepCut_resnet50_ratmovementNov5shuffle1_22500'], ['body', 'head', 'neck', 'rear', 'tail'], ['likelihood', 'x', 'y']],
        #                   labels=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3, 4, 4, 4], [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
        #                   names=['scorer', 'bodyparts', 'coords'])>


        #df['DeepCut_resnet50_ratmovementNov5shuffle1_22500']['body'][['x', 'y', 'likelihood']]

        # Convert multiindex data frame to numpy array
        o = df.values

        # add a preprocessing function here to sort data based on likelihood
        # ...

        # remove likelihood column
        m = np.delete(o, np.arange(feature_coords-1, nfeatures*feature_coords , feature_coords), axis=1)

        do_pca(m)

