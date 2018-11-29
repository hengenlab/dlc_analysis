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
from scipy import spatial

# Calculate features

# calculate distance
def calc_dist(data):
    dist_head = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data[:,0:2],'euclidean')),1)
    dist_neck = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data[:,2:4],'euclidean')),1)
    dist_body = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data[:,4:6],'euclidean')),1)
    dist_rear = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data[:,6:8],'euclidean')),1)
    dist_tail = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data[:,8:10],'euclidean')),1)
    distmat = np.transpose([dist_head,dist_neck,dist_body,dist_rear,dist_tail])
    return distmat

# calculate velocity
def calc_velocity(distmat):
    nframes = 9000
    fps = 15
    vidduration = nframes/fps
    timestep = vidduration/nframes
    velmat = distmat/timestep
    return velmat

# calculate acceleration
def calc_acceleration(velmat):
    #make a matrix that is shifted vertically by 1
    rollmat = np.roll(velmat,1,axis=0)
    #make first row of shifted matrix 0 since it is the initial point in time
    rollmat[0] = 0
    #subtract the original velocity matrix from the shifted matrix
    accelmat = velmat-rollmat
    return accelmat

# calculate logical array 0000 or 11111, if animal moved in last 10 sec move to 1 else 0
def calc_relative_motion():
    ...

# Temporal what was happening 5 min earlier
def calc_previous_state():
    tol_val = 0.01
    ...


# calculate distance relative to body of animal
def calc_dist_from_body(data):
    pos_head = data[:,0:2] - data[:,4:6]
    pos_neck = data[:,2:4] - data[:,4:6]
    pos_body = data[:,4:6] - data[:,4:6]
    pos_rear = data[:,6:8] - data[:,4:6]
    pos_tail = data[:,8:10] - data[:,4:6]
    distfrombody = np.hstack((pos_head,pos_neck,pos_body,pos_rear,pos_tail))
    return distfrombody

def do_lda(data):
    ...

def do_tsnei(data, ncomponents, verbosity, iperplexity, maxiter):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances

    distance_matrix = pairwise_distances(data, data, metric='cosine', n_jobs=-1)

    tsne = TSNE(n_components=ncomponents, verbose=verbosity, perplexity=iperplexity, n_iter=maxiter, metric='precomputed')
    #tsne_results = tsne.fit_transform(data)
    tsne_results = tsne.fit_transform(distance_matrix)
    return tsne_results

def do_pca(data):
    mu = data.mean(axis=0)
    data = data - mu
    data = (data - mu)/data.std(axis=0)  # Uncommenting this reproduces mlab.PCA results
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    print(eigenvectors)
    return projected_data, eigenvectors, eigenvalues, V, sigma

def plot_tsne_out(out_tsne):
    from matplotlib import pyplot as plt
    plt.scatter(out_tsne[:,0], out_tsne[:,1])
    plt.show()

def plot_tsne_out_3d(out_tsne):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = out_tsne[:,0]
    y = out_tsne[:,1]
    z = out_tsne[:,2]

    ax.scatter(x, y, z, c='r', marker='o')
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.show()


def do_kmeans(data, k):
    from sklearn.cluster import KMeans
    #seed = 0
    #km = KMeans(n_clusters=k, 'random', n_init=10, max_iter=1000, random_state=seed)
    km = KMeans(n_clusters=k, n_init=100, max_iter=10000, tol=1e-6)
    km.fit(data)
    y_cluster_kmeans = km.predict(data)
    return y_cluster_kmeans


def plot_with_labels(out_tsne, label):
        import numpy as np
        from matplotlib import pyplot as plt
        x = out_tsne[:,0]
        y = out_tsne[:,1]
        
        fig, ax = plt.subplots()
        for l in np.unique(label):
             i = np.where(label == l)
             ax.scatter(x[i], y[i], [], label = l)
             
        #ax.legend()
        plt.show()



if __name__ == '__main__':
    from sklearn.decomposition import PCA
    # Constants
    nfeatures = 5
    feature_coords = 3
    lcal = 0
    lsave = 1
    
    # open file browser
    if len(sys.argv) > 1:
        dataname = (sys.argv[1])
    else:
        Tk().withdraw() 
        dataname = askopenfilename() # open file browser
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
    
    
    #get features for PCA/LDA
    if lcal == 1:
        distmat = calc_dist(m)
        if lsave == 1:
            np.save('distmat.npy', distmat)
        velmat = calc_velocity(distmat)
        if lsave == 1:
            np.save('velmat.npy', velmat)
        accelmat = calc_acceleration(velmat)
        if lsave == 1:
            np.save('accelmat.npy', accelmat)
        #distfrombody = calc_dist_from_body(m)
    else:
        distmat = np.load('distmat.npy')
        velmat = np.load('velmat.npy')
        accelmat = np.load('accelmat.npy')

    
    # Feature matrix
    feature_mat = np.hstack((distmat, velmat, accelmat))
    print(feature_mat.shape)
    
    # kmeans for labels
    labels = do_kmeans(feature_mat, 4)
    if lsave == 1:
        np.save('labels.npy', labels)
    
    # do pca
    #reduced_data = PCA(n_components=15).fit_transform(feature_mat)
    proj_data, eig_vec, eig_val, V, sigma  = do_pca(feature_mat)
    #plot_with_labels(proj_data, labels)
    #proj_data, eigenvectors, eigenvalues, V, sigma = do_pca(feature_mat)
    
    # t-sne
    # def do_tsnei(data, ncomponents, verbosity, iperplexity, maxiter):
    r_tsne = do_tsnei(proj_data, 2, 1, 40, 1000)
    if lsave == 1:
        np.save('r_tsne.npy', r_tsne)
    
    # plot
    # plot_tsne_out(r_tsne)
    # plot 3d plot_tsne_out_3d(r_tsne)
    plot_with_labels(r_tsne, labels)

